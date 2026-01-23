"""Main pipeline orchestration for news analysis.

This module coordinates the entire news analysis workflow:

Pipeline Flow (v2 - Grounded):
    1. FETCH: Concurrently fetch all RSS feeds
    2. DEDUP: Filter out previously-seen stories (by hash)
    3. CLASSIFY: Run classifier agent (local LLM) on new stories
    4. SPLIT: Separate important from non-important stories
    5. CONTEXT: For important stories, gather context in parallel:
       - URL Fetch: Get full article content
       - Hybrid RAG: Query related stories from database
    6. ANALYZE: Run researcher agent (Gemini + grounding) with pre-fetched context
    7. SAVE: Store all results in database + embeddings for important stories
    8. NOTIFY: Send reports and alerts for important stories

Key Changes from v1:
    - Researcher uses Gemini with Google Search grounding (no custom tools)
    - Context (article + related stories) pre-fetched before researcher call
    - Hybrid RAG: BM25 + Vector search with RRF fusion
    - Single Gemini API call per story (vs 1-15 with tool round-trips)
    - Embeddings stored for important stories (for future vector search)
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
from models.story import Story
from models.classification import ClassificationResult
from agents.classifier import ClassifierAgent
from agents.researcher import ResearcherAgent, StoryContext
from tools.fetch import fetch_article
from embeddings import encode_text
from notifications import notify_batch
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


async def _gather_story_context(
    story: Story,
    classification: ClassificationResult | None,
    db: Database,
) -> StoryContext:
    """Gather all context for a story in parallel.

    Runs URL fetch and hybrid RAG search concurrently, then
    combines results into a StoryContext for the researcher.

    Args:
        story: Story to gather context for
        classification: Classification result (for category context)
        db: Database instance for hybrid RAG search

    Returns:
        StoryContext with all gathered information
    """
    # Build search query from title + description snippet
    query_text = story.title
    if story.description:
        # Add first 200 chars of description for better search
        desc_snippet = story.description[:200].replace('\n', ' ')
        query_text = f"{story.title} {desc_snippet}"

    # Generate embedding for vector search
    try:
        query_embedding = encode_text(query_text)
    except Exception as e:
        logger.warning("Embedding generation failed: %s", e)
        query_embedding = None

    # Run fetch and RAG in parallel
    async def fetch_content() -> tuple[str, bool]:
        if story.article_url:
            result = await fetch_article(story.article_url)
            if result.success:
                logger.debug(
                    "Article fetched | url=%s chars=%d",
                    story.article_url[:50], len(result.content)
                )
                return result.content, True
            logger.debug("Article fetch failed for %s: %s", story.article_url[:50], result.error)
        return "", False

    async def query_rag() -> tuple[str, int]:
        try:
            results = db.hybrid_search(
                query=story.title,
                query_embedding=query_embedding,
                limit=5,
                days=30,
            )
            count = len(results.stories)
            logger.debug(
                "RAG search complete | query='%s' results=%d",
                story.title[:40], count
            )
            return results.format_context(), count
        except Exception as e:
            logger.warning("Hybrid RAG search failed: %s", e)
            return "No related stories found in database.", 0

    # Execute in parallel
    (article_content, article_ok), (related_stories, rag_count) = await asyncio.gather(
        fetch_content(),
        query_rag(),
    )

    logger.debug(
        "Context gathered | title='%s' article=%s rag_results=%d",
        story.title[:40], "yes" if article_ok else "no", rag_count
    )

    return StoryContext(
        story=story,
        classification=classification,
        article_content=article_content,
        related_stories=related_stories,
    )


class Pipeline:
    """Async news analysis pipeline using PydanticAI agents.

    Orchestrates the complete workflow from RSS fetching through
    analysis and notification. Manages all component lifecycles.

    Components:
        - Database: SQLite storage with hybrid RAG (FTS5 + embeddings)
        - ClassifierAgent: Fast importance classification (local LLM)
        - ResearcherAgent: Deep analysis with Gemini + grounding

    Architecture:
        1. Classifier runs on local LLM (fast, cheap)
        2. For important stories, context gathered in parallel (deterministic)
        3. Researcher runs with pre-fetched context + Google Search grounding
        4. Single Gemini API call per story
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
            4. Classify new stories for importance (local LLM)
            5. Gather context for important stories (parallel fetch + RAG)
            6. Analyze important stories (Gemini + grounding)
            7. Save all results + embeddings to database
            8. Send notifications for important stories

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

            # Apply max_stories limit BEFORE classification
            if max_stories > 0 and len(new_stories) > max_stories:
                new_stories.sort(key=lambda s: s.pub_date, reverse=True)
                new_stories = new_stories[:max_stories]
                logger.info("Limited to %d latest stories for processing", max_stories)

            # Classify (local LLM)
            classified = await self.classifier.classify_batch(
                new_stories,
                max_concurrent=self.config.max_workers,
            )
            stats.classified = len(classified)

            important = [(s, c) for s, c in classified if c.is_important]
            not_important = [(s, c) for s, c in classified if not c.is_important]
            stats.important = len(important)

            logger.info("Classification complete | important=%d skip=%d", len(important), len(not_important))

            # Save non-important stories (no embedding needed)
            for story, _ in not_important:
                self.db.save(story, Analysis.empty(), commit=False)

            # Process important stories
            if important:
                # Gather context for all important stories in parallel
                logger.info("Gathering context for %d important stories", len(important))
                context_tasks = [
                    _gather_story_context(story, classification, self.db)
                    for story, classification in important
                ]
                story_contexts = await asyncio.gather(*context_tasks)

                # Analyze with researcher (Gemini + grounding)
                analyzed = await self.researcher.analyze_batch(
                    story_contexts,
                    max_concurrent=min(3, self.config.max_workers),
                )
                stats.analyzed = len(analyzed)

                # Save results and generate embeddings
                to_notify = []
                for story, report in analyzed:
                    if report.summary:
                        analysis = Analysis.from_report(report)
                        self.db.save(story, analysis, commit=False)

                        # Generate and save embedding for important story
                        try:
                            # Embed title + summary for high-quality vector search
                            embed_text = f"{story.title} {report.summary[:500]}"
                            embedding = encode_text(embed_text)
                            self.db.save_embedding(story.hash, embedding, commit=False)
                        except Exception as e:
                            logger.warning("Embedding save failed for %s: %s", story.hash, e)

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

            # Commit all changes
            self.db.commit()

        except asyncio.CancelledError:
            logger.info("Pipeline run cancelled")
            raise
        except Exception as e:
            logger.error("Pipeline error | type=%s error=%s", type(e).__name__, e, exc_info=True)
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
