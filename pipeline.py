"""Main pipeline orchestration for news analysis.

This module coordinates the entire news analysis workflow:

Pipeline Flow:
    1. FETCH: Concurrently fetch all RSS feeds
    2. DEDUP: Filter out previously-seen stories (by hash)
    3. CLASSIFY: Run classifier agent on new stories
    4. SPLIT: Separate important from non-important stories
    5. ANALYZE: Run researcher agent on important stories
    6. SAVE: Store all results in database
    7. NOTIFY: Send reports and alerts for important stories

The Pipeline class manages all components and provides both
single-run and continuous polling modes.

Usage:
    >>> config = Config.load()
    >>> pipeline = Pipeline(config)
    >>> stats = await pipeline.run_once()
    >>> pipeline.close()
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

from config import Config
from database import Database
from feeds import fetch_all_feeds
from models.research import Analysis
from agents.classifier import ClassifierAgent
from agents.researcher import ResearcherAgent
from notifications import notify_batch
from memory.vector_store import VectorStore
from observability.logging import set_run_context, clear_context

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a single pipeline run.

    Tracks counts at each stage of the pipeline for monitoring
    and debugging purposes.

    Attributes:
        fetched: Total stories fetched from feeds
        skipped: Stories already in database (deduplicated)
        classified: Stories processed by classifier
        important: Stories marked important
        analyzed: Stories processed by researcher
        notified: Successful notifications sent
        errors: Count of errors at any stage
        duration: Total run time in seconds
    """

    fetched: int = 0      # Stories from RSS feeds
    skipped: int = 0      # Already seen (dedup)
    classified: int = 0   # Processed by classifier
    important: int = 0    # Marked important
    analyzed: int = 0     # Processed by researcher
    notified: int = 0     # Notifications sent
    errors: int = 0       # Errors encountered
    duration: float = 0.0 # Run time (seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["duration"] = round(d["duration"], 2)
        return d


class Pipeline:
    """Async news analysis pipeline using PydanticAI agents.

    Orchestrates the complete workflow from RSS fetching through
    analysis and notification. Manages all component lifecycles.

    Components:
        - Database: SQLite storage for dedup and results
        - ClassifierAgent: Fast importance classification
        - ResearcherAgent: Deep analysis with tools
        - VectorStore: Optional semantic search (if enabled)

    Modes:
        - run_once(): Single execution
        - run_continuous(): Polling loop with configurable interval
    """

    def __init__(self, config: Config):
        """Initialize pipeline with all components.

        Args:
            config: Application configuration
        """
        self.config = config
        self.db = Database(config.db_path)
        self.classifier = ClassifierAgent(config)
        self.researcher = ResearcherAgent(config)
        self.vector_store: VectorStore | None = None

        # Optional: Vector memory for semantic search
        if config.enable_memory:
            self.vector_store = VectorStore(config.vector_db_path)

        # Optional: Distributed tracing
        if config.enable_logfire:
            from observability.tracing import setup_tracing
            setup_tracing(enabled=True, service_name="photon", token=config.logfire_token)

    async def run_once(self, max_stories: int = 0) -> PipelineStats:
        """Execute one complete pipeline run.

        Steps:
            1. Prune old records from database
            2. Fetch all RSS feeds concurrently
            3. Deduplicate against existing stories
            4. Classify new stories for importance
            5. Analyze important stories in depth (limited by max_stories)
            6. Save all results to database
            7. Send notifications for important stories

        Args:
            max_stories: Maximum important stories to analyze (0 = unlimited).
                         When limited, selects the latest stories by pub_date.

        Returns:
            PipelineStats with counts from each stage
        """
        run_id = uuid.uuid4().hex[:8]
        set_run_context(run_id)
        start = time.time()
        stats = PipelineStats()

        logger.info("Pipeline started | feeds=%d", len(self.config.rss_urls))

        try:
            # Prune old records
            pruned = self.db.prune(self.config.prune_after_days)
            if pruned:
                logger.debug("Pruned old records | count=%d", pruned)

            # Fetch stories
            stories = await fetch_all_feeds(
                self.config.rss_urls,
                self.config.max_age_hours,
                max_concurrent=self.config.max_workers,
            )
            stats.fetched = len(stories)

            # Deduplicate
            seen = self.db.seen_hashes({s.hash for s in stories})
            new_stories = [s for s in stories if s.hash not in seen]
            stats.skipped = stats.fetched - len(new_stories)

            logger.info("Fetch complete | total=%d new=%d", stats.fetched, len(new_stories))

            if not new_stories:
                stats.duration = time.time() - start
                return stats

            # Classify
            classified = await self.classifier.classify_batch(
                new_stories,
                max_concurrent=self.config.max_workers,
            )
            stats.classified = len(classified)

            important = [(s, c) for s, c in classified if c.is_important]
            not_important = [(s, c) for s, c in classified if not c.is_important]
            stats.important = len(important)

            # Limit important stories if max_stories is set
            if max_stories > 0 and len(important) > max_stories:
                # Sort by pub_date descending (latest first) and take top N
                important.sort(key=lambda x: x[0].pub_date, reverse=True)
                skipped_important = important[max_stories:]
                important = important[:max_stories]
                # Move skipped important stories to not_important (save without analysis)
                not_important.extend(skipped_important)
                logger.info("Limited to %d latest important stories (skipped %d)", max_stories, len(skipped_important))

            logger.info("Classification complete | important=%d skip=%d", len(important), len(not_important))

            # Save non-important stories
            for story, _ in not_important:
                self.db.save(story, Analysis.empty(), commit=False)

            # Analyze important stories
            if important:
                analyzed = await self.researcher.analyze_batch(
                    important,
                    max_concurrent=min(3, self.config.max_workers),
                )
                stats.analyzed = len(analyzed)

                to_notify = []
                for story, report in analyzed:
                    if report.summary:
                        analysis = Analysis.from_report(report)
                        self.db.save(story, analysis, commit=False)

                        if self.vector_store:
                            await self.vector_store.add_story(
                                story.hash, story.title, report.summary,
                                story.pub_date, story.source_url,
                            )

                        to_notify.append((story, analysis))
                        logger.info("Story analyzed | hash=%s title=%s", story.hash, story.title[:60])
                    else:
                        stats.errors += 1
                        self.db.save(story, Analysis.empty(), commit=False)
                        logger.warning("Empty analysis result | hash=%s", story.hash)

                # Send notifications
                if to_notify:
                    ok, fail = await notify_batch(to_notify, self.config)
                    stats.notified = ok
                    stats.errors += fail

            # Commit
            self.db.commit()
            if self.vector_store:
                self.vector_store.persist()

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            stats.errors += 1

        stats.duration = time.time() - start
        logger.info(
            "Pipeline done | duration=%.1fs important=%d notified=%d errors=%d",
            stats.duration, stats.important, stats.notified, stats.errors
        )
        clear_context()
        return stats

    async def run_continuous(self, max_stories: int = 0) -> None:
        """Run pipeline continuously with polling.

        Args:
            max_stories: Maximum important stories to analyze per run (0 = unlimited).
        """
        run_count = 0
        total_important = 0
        total_errors = 0

        logger.info("Starting continuous mode | interval=%ds", self.config.poll_interval_seconds)

        try:
            while True:
                run_count += 1
                try:
                    stats = await self.run_once(max_stories=max_stories)
                    total_important += stats.important
                    total_errors += stats.errors
                except Exception as e:
                    logger.error("Run failed | run=%d error=%s", run_count, e, exc_info=True)
                    total_errors += 1

                logger.info("Run complete | run=%d total_important=%d total_errors=%d", run_count, total_important, total_errors)
                await asyncio.sleep(self.config.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Pipeline stopped | runs=%d total_important=%d total_errors=%d", run_count, total_important, total_errors)
            raise

    def close(self) -> None:
        """Clean up resources."""
        self.db.close()
        if self.vector_store:
            self.vector_store.persist()


async def run_once(config: Config, max_stories: int = 0) -> dict[str, Any]:
    """Run pipeline once and return stats dict.

    Args:
        config: Application configuration
        max_stories: Maximum important stories to analyze (0 = unlimited)
    """
    pipeline = Pipeline(config)
    try:
        return (await pipeline.run_once(max_stories=max_stories)).to_dict()
    finally:
        pipeline.close()


async def run_continuous(config: Config, max_stories: int = 0) -> None:
    """Run pipeline continuously.

    Args:
        config: Application configuration
        max_stories: Maximum important stories to analyze per run (0 = unlimited)
    """
    pipeline = Pipeline(config)
    try:
        await pipeline.run_continuous(max_stories=max_stories)
    finally:
        pipeline.close()
