"""Main pipeline orchestration for news analysis."""

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

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    fetched: int = 0
    skipped: int = 0
    classified: int = 0
    important: int = 0
    analyzed: int = 0
    notified: int = 0
    errors: int = 0
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(d["duration"], 2)
        return d


class Pipeline:
    """Async news analysis pipeline using PydanticAI agents."""

    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_path)
        self.classifier = ClassifierAgent(config)
        self.researcher = ResearcherAgent(config)
        self.vector_store: VectorStore | None = None

        if config.enable_memory:
            self.vector_store = VectorStore(config.vector_db_path)

        if config.enable_logfire:
            from observability.tracing import setup_tracing
            setup_tracing(enabled=True, service_name="photon", token=config.logfire_token)

    async def run_once(self) -> PipelineStats:
        """Execute one pipeline run."""
        run_id = uuid.uuid4().hex[:8]
        start = time.time()
        stats = PipelineStats()

        logger.info(f"Pipeline started | run={run_id} feeds={len(self.config.rss_urls)}")

        try:
            # Prune old records
            pruned = self.db.prune(self.config.prune_after_days)
            if pruned:
                logger.debug(f"Pruned {pruned} old records")

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

            logger.info(f"Fetched {stats.fetched} stories, {len(new_stories)} new")

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

            logger.info(f"Classified: {stats.important} important, {len(not_important)} skip")

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
                        logger.info(f"IMPORTANT [{story.hash}] {story.title[:60]}")
                    else:
                        stats.errors += 1
                        self.db.save(story, Analysis.empty(), commit=False)
                        logger.warning(f"Empty analysis [{story.hash}]")

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
            f"Pipeline done in {stats.duration:.1f}s | "
            f"important={stats.important} notified={stats.notified} errors={stats.errors}"
        )
        return stats

    async def run_continuous(self) -> None:
        """Run pipeline continuously with polling."""
        run_count = 0
        total_important = 0
        total_errors = 0

        logger.info(f"Starting continuous mode | interval={self.config.poll_interval_seconds}s")

        try:
            while True:
                run_count += 1
                try:
                    stats = await self.run_once()
                    total_important += stats.important
                    total_errors += stats.errors
                except Exception as e:
                    logger.error(f"Run #{run_count} failed: {e}")
                    total_errors += 1

                logger.info(f"Run #{run_count} | total: important={total_important} errors={total_errors}")
                await asyncio.sleep(self.config.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info(f"Stopped after {run_count} runs | important={total_important} errors={total_errors}")
            raise

    def close(self) -> None:
        """Clean up resources."""
        self.db.close()
        if self.vector_store:
            self.vector_store.persist()


async def run_once(config: Config) -> dict[str, Any]:
    """Run pipeline once and return stats dict."""
    pipeline = Pipeline(config)
    try:
        return (await pipeline.run_once()).to_dict()
    finally:
        pipeline.close()


async def run_continuous(config: Config) -> None:
    """Run pipeline continuously."""
    pipeline = Pipeline(config)
    try:
        await pipeline.run_continuous()
    finally:
        pipeline.close()
