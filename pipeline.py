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
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Config
from database import Database
from feeds import fetch_all_feeds
from models.research import Analysis, ResearchReport
from models.story import Story
from models.classification import ClassificationResult
from agents.classifier import ClassifierAgent
from agents.researcher import ResearcherAgent, StoryContext
from tools.fetch import fetch_article
from embeddings import encode_text
from notifications import notify_batch
from observability.logging import set_run_context, clear_context

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_URL_PATTERN = re.compile(r"https?://\S+")
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\n\r\t]')


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
        input_tokens: Total input tokens used by researcher
        output_tokens: Total output tokens used by researcher
        articles_fetched: Number of articles successfully fetched
        rag_queries: Number of RAG queries performed
    """

    fetched: int = 0      # Stories from RSS feeds
    skipped: int = 0      # Already seen (dedup)
    classified: int = 0   # Processed by classifier
    important: int = 0    # Marked important
    analyzed: int = 0     # Processed by researcher
    notified: int = 0     # Notifications sent
    errors: int = 0       # Errors encountered
    duration: float = 0.0 # Run time (seconds)
    input_tokens: int = 0  # Researcher input tokens
    output_tokens: int = 0 # Researcher output tokens
    articles_fetched: int = 0  # Articles successfully fetched
    rag_queries: int = 0   # RAG queries performed

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
            if result.success and result.content:
                logger.debug(
                    "Article fetched | url=%s chars=%d",
                    story.article_url[:50], len(result.content)
                )
                return result.content, True
            # Log why fetch failed
            if not result.success:
                logger.debug("Article fetch failed | url=%s error=%s", story.article_url[:50], result.error)
            elif not result.content:
                logger.debug("Article fetch empty | url=%s", story.article_url[:50])
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

                # Log context gathering summary
                articles_ok = sum(1 for ctx in story_contexts if ctx.article_content)
                rag_ok = sum(1 for ctx in story_contexts if ctx.related_stories and "No related" not in ctx.related_stories)
                stats.articles_fetched = articles_ok
                stats.rag_queries = len(story_contexts)
                logger.info(
                    "Context gathered | stories=%d articles=%d/%d rag=%d/%d",
                    len(story_contexts), articles_ok, len(story_contexts), rag_ok, len(story_contexts)
                )

                # Analyze with researcher (Gemini + grounding)
                analyzed, input_tokens, output_tokens = await self.researcher.analyze_batch(
                    story_contexts,
                    max_concurrent=min(3, self.config.max_workers),
                )
                stats.analyzed = len(analyzed)
                stats.input_tokens = input_tokens
                stats.output_tokens = output_tokens

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
            "Pipeline done | duration=%.1fs important=%d notified=%d errors=%d tokens=%d/%d",
            stats.duration, stats.important, stats.notified, stats.errors,
            stats.input_tokens, stats.output_tokens
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


def _report_metrics(report: ResearchReport) -> dict[str, Any]:
    """Compute lightweight quality metrics for a research report."""
    summary = report.summary or ""
    thought = report.thought or ""
    key_points_text = "\n".join(report.key_points or [])

    citations = _CITATION_PATTERN.findall(summary + "\n" + key_points_text)
    sources = _URL_PATTERN.findall(thought)

    return {
        "summary_chars": len(summary),
        "summary_words": len(summary.split()),
        "thought_chars": len(thought),
        "key_points_count": len(report.key_points or []),
        "related_topics_count": len(report.related_topics or []),
        "citations_total": len(citations),
        "citations_unique": len(set(citations)),
        "sources_total": len(sources),
        "sources_unique": len(set(sources)),
    }


def _render_report_markdown(
    story: Story,
    classification: ClassificationResult | None,
    report: ResearchReport,
    model_name: str,
) -> str:
    """Render a full markdown report for comparison runs."""
    pub_date_str = story.pub_date.strftime("%Y-%m-%d %H:%M")
    analyzed_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {story.title}",
        "",
        f"**Model:** {model_name}",
        f"**Published:** {pub_date_str}",
        f"**Source:** {story.source_url}",
    ]
    if story.article_url:
        lines.append(f"**Article URL:** {story.article_url}")
    lines.append(f"**Analyzed:** {analyzed_str}")

    if classification:
        lines.extend([
            "",
            "## Classification",
            "",
            f"- Important: {classification.is_important}",
            f"- Category: {classification.category.value}",
            f"- Confidence: {classification.confidence:.2f}",
        ])
        if classification.reasoning:
            lines.append(f"- Reasoning: {classification.reasoning}")

    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
        report.summary or "",
    ])

    if report.thought:
        lines.extend([
            "",
            "---",
            "",
            "## Thought",
            "",
            report.thought,
        ])

    if report.key_points:
        lines.extend([
            "",
            "---",
            "",
            "## Key Points",
            "",
        ])
        lines.extend([f"- {point}" for point in report.key_points])

    if report.related_topics:
        lines.extend([
            "",
            "---",
            "",
            "## Related Topics",
            "",
        ])
        lines.extend([f"- {topic}" for topic in report.related_topics])

    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_title(title: str, max_length: int = 80) -> str:
    """Sanitize story title for filenames."""
    s = _INVALID_FILENAME_CHARS.sub(" ", title)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_length:
        s = s[:max_length].rstrip()
    return s or "report"


def _render_comparison_markdown(
    story: Story,
    classification: ClassificationResult | None,
    flash_report: ResearchReport,
    pro_report: ResearchReport,
    flash_model: str,
    pro_model: str,
    flash_metrics: dict[str, Any],
    pro_metrics: dict[str, Any],
) -> str:
    """Render a side-by-side markdown comparison for a single story."""
    pub_date_str = story.pub_date.strftime("%Y-%m-%d %H:%M")
    analyzed_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {story.title}",
        "",
        "**Model:** Comparison",
        f"**Published:** {pub_date_str}",
        f"**Source:** {story.source_url}",
    ]
    if story.article_url:
        lines.append(f"**Article URL:** {story.article_url}")
    lines.append(f"**Analyzed:** {analyzed_str}")

    if classification:
        lines.extend([
            "",
            "## Classification",
            "",
            f"- Important: {classification.is_important}",
            f"- Category: {classification.category.value}",
            f"- Confidence: {classification.confidence:.2f}",
        ])
        if classification.reasoning:
            lines.append(f"- Reasoning: {classification.reasoning}")

    lines.extend([
        "",
        "---",
        "",
        "## Flash Report",
        "",
        f"**Model:** {flash_model}",
        "",
        "### Summary",
        "",
        flash_report.summary or "",
    ])
    if flash_report.thought:
        lines.extend(["", "### Thought", "", flash_report.thought])
    if flash_report.key_points:
        lines.extend(["", "### Key Points", ""])
        lines.extend([f"- {point}" for point in flash_report.key_points])
    if flash_report.related_topics:
        lines.extend(["", "### Related Topics", ""])
        lines.extend([f"- {topic}" for topic in flash_report.related_topics])

    lines.extend([
        "",
        "---",
        "",
        "## Pro Report",
        "",
        f"**Model:** {pro_model}",
        "",
        "### Summary",
        "",
        pro_report.summary or "",
    ])
    if pro_report.thought:
        lines.extend(["", "### Thought", "", pro_report.thought])
    if pro_report.key_points:
        lines.extend(["", "### Key Points", ""])
        lines.extend([f"- {point}" for point in pro_report.key_points])
    if pro_report.related_topics:
        lines.extend(["", "### Related Topics", ""])
        lines.extend([f"- {topic}" for topic in pro_report.related_topics])

    lines.extend([
        "",
        "---",
        "",
        "## Metrics",
        "",
        "| Metric | Flash | Pro |",
        "| --- | ---: | ---: |",
        f"| Summary chars | {flash_metrics['summary_chars']} | {pro_metrics['summary_chars']} |",
        f"| Summary words | {flash_metrics['summary_words']} | {pro_metrics['summary_words']} |",
        f"| Citations | {flash_metrics['citations_total']} | {pro_metrics['citations_total']} |",
        f"| Sources | {flash_metrics['sources_total']} | {pro_metrics['sources_total']} |",
        f"| Key points | {flash_metrics['key_points_count']} | {pro_metrics['key_points_count']} |",
        f"| Related topics | {flash_metrics['related_topics_count']} | {pro_metrics['related_topics_count']} |",
    ])

    return "\n".join(lines)


async def compare_models(
    config: Config,
    limit: int = 10,
    output_dir: Path | None = None,
    max_concurrent: int = 2,
) -> dict[str, Any]:
    """Compare flash vs pro researcher outputs on a sample of stories.

    Args:
        config: Application configuration
        limit: Number of stories to analyze
        output_dir: Directory for comparison reports (default: reports/compare/<timestamp>)
        max_concurrent: Max concurrent story comparisons

    Returns:
        Summary dict with run metadata and output location
    """
    start = time.time()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = output_dir or (config.reports_dir / "compare" / run_stamp)
    base_dir.mkdir(parents=True, exist_ok=True)

    flash_model = config.researcher_model
    pro_model = config.researcher_model_pro

    # Initialize agents
    classifier = ClassifierAgent(config)
    flash_agent = ResearcherAgent(config)
    pro_config = replace(config, researcher_model=pro_model)
    pro_agent = ResearcherAgent(pro_config)

    with Database(config.db_path) as db:
        # Fetch stories
        stories = await fetch_all_feeds(
            config.rss_urls,
            config.max_age_hours,
            max_concurrent=config.max_workers,
        )

        if not stories:
            return {
                "run_dir": str(base_dir),
                "stories_total": 0,
                "stories_selected": 0,
                "analyzed": 0,
                "models": {"flash": flash_model, "pro": pro_model},
                "duration_seconds": round(time.time() - start, 2),
            }

        # Prefer new stories, but fall back to recent if not enough
        seen = db.seen_hashes({s.hash for s in stories})
        new_stories = [s for s in stories if s.hash not in seen]
        if len(new_stories) < limit:
            logger.info(
                "Not enough new stories (%d) for limit=%d; using latest from feeds",
                len(new_stories), limit
            )
            candidate_stories = stories
        else:
            candidate_stories = new_stories

        candidate_stories.sort(key=lambda s: s.pub_date, reverse=True)
        selected_stories = candidate_stories[:limit]

        # Classify selected stories
        classified = await classifier.classify_batch(
            selected_stories,
            max_concurrent=config.max_workers,
        )

        # Gather context for each story
        context_tasks = [
            _gather_story_context(story, classification, db)
            for story, classification in classified
        ]
        story_contexts = await asyncio.gather(*context_tasks)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_pair(
        index: int,
        story_ctx: StoryContext,
    ) -> dict[str, Any]:
        async with semaphore:
            flash_task = flash_agent.analyze(story_ctx)
            pro_task = pro_agent.analyze(story_ctx)
            (flash_report, flash_in, flash_out), (pro_report, pro_in, pro_out) = await asyncio.gather(
                flash_task, pro_task, return_exceptions=False
            )

        flash_metrics = _report_metrics(flash_report)
        pro_metrics = _report_metrics(pro_report)

        delta = {}
        for key, flash_val in flash_metrics.items():
            pro_val = pro_metrics.get(key)
            if isinstance(flash_val, (int, float)) and isinstance(pro_val, (int, float)):
                delta[key] = pro_val - flash_val

        return {
            "index": index,
            "story": story_ctx.story,
            "classification": story_ctx.classification,
            "flash": {
                "model": flash_model,
                "report": flash_report,
                "input_tokens": flash_in,
                "output_tokens": flash_out,
                "metrics": flash_metrics,
            },
            "pro": {
                "model": pro_model,
                "report": pro_report,
                "input_tokens": pro_in,
                "output_tokens": pro_out,
                "metrics": pro_metrics,
            },
            "delta": delta,
        }

    async def save_item(item: dict[str, Any]) -> dict[str, Any]:
        story = item["story"]
        classification = item["classification"]
        story_dir = base_dir / f"{item['index']:02d}_{story.hash}"
        story_dir.mkdir(parents=True, exist_ok=True)
        title_slug = _sanitize_title(story.title)

        flash_report = item["flash"]["report"]
        pro_report = item["pro"]["report"]

        flash_md = _render_report_markdown(story, classification, flash_report, flash_model)
        pro_md = _render_report_markdown(story, classification, pro_report, pro_model)
        compare_md = _render_comparison_markdown(
            story,
            classification,
            flash_report,
            pro_report,
            flash_model,
            pro_model,
            item["flash"]["metrics"],
            item["pro"]["metrics"],
        )

        (story_dir / "flash.md").write_text(flash_md, encoding="utf-8")
        (story_dir / "pro.md").write_text(pro_md, encoding="utf-8")
        (story_dir / "compare.md").write_text(compare_md, encoding="utf-8")

        # Also save flat copies directly under reports/ for easy inspection.
        flat_prefix = f"{run_stamp}_{item['index']:02d}_{title_slug}"
        (config.reports_dir / f"{flat_prefix}_flash.md").write_text(flash_md, encoding="utf-8")
        (config.reports_dir / f"{flat_prefix}_pro.md").write_text(pro_md, encoding="utf-8")

        _write_json(
            story_dir / "flash.json",
            {
                "model": flash_model,
                "story": story.model_dump(mode="json"),
                "classification": classification.model_dump(mode="json") if classification else None,
                "report": flash_report.model_dump(mode="json"),
                "usage": {
                    "input_tokens": item["flash"]["input_tokens"],
                    "output_tokens": item["flash"]["output_tokens"],
                },
                "metrics": item["flash"]["metrics"],
            },
        )
        _write_json(
            story_dir / "pro.json",
            {
                "model": pro_model,
                "story": story.model_dump(mode="json"),
                "classification": classification.model_dump(mode="json") if classification else None,
                "report": pro_report.model_dump(mode="json"),
                "usage": {
                    "input_tokens": item["pro"]["input_tokens"],
                    "output_tokens": item["pro"]["output_tokens"],
                },
                "metrics": item["pro"]["metrics"],
            },
        )
        _write_json(
            story_dir / "metrics.json",
            {
                "story": {
                    "hash": story.hash,
                    "title": story.title,
                    "pub_date": story.pub_date.isoformat(),
                    "source_url": story.source_url,
                    "article_url": story.article_url,
                },
                "flash_metrics": item["flash"]["metrics"],
                "pro_metrics": item["pro"]["metrics"],
                "delta": item["delta"],
            },
        )

        return {
            "index": item["index"],
            "hash": story.hash,
            "title": story.title,
            "folder": story_dir.name,
            "flash_metrics": item["flash"]["metrics"],
            "pro_metrics": item["pro"]["metrics"],
            "delta": item["delta"],
        }

    # Run and persist as each story completes
    tasks = [asyncio.create_task(analyze_pair(i + 1, ctx)) for i, ctx in enumerate(story_contexts)]
    results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    total = len(tasks)
    completed = 0
    for coro in asyncio.as_completed(tasks):
        item = await coro
        row = await save_item(item)
        comparison_rows.append(row)
        results.append(item)
        completed += 1
        logger.info(
            "Comparison saved | %d/%d | title=%s",
            completed, total, item["story"].title[:60]
        )

    comparison_rows.sort(key=lambda r: r["index"])
    results.sort(key=lambda r: r["index"])

    # Save summary files
    summary = {
        "run_dir": str(base_dir),
        "run_stamp": run_stamp,
        "models": {"flash": flash_model, "pro": pro_model},
        "stories_total": len(stories),
        "stories_selected": len(story_contexts),
        "duration_seconds": round(time.time() - start, 2),
        "rows": comparison_rows,
    }
    _write_json(base_dir / "comparison.json", summary)

    # Simple markdown table for quick scan
    table_lines = [
        "# Researcher Comparison",
        "",
        f"- Flash: {flash_model}",
        f"- Pro: {pro_model}",
        "",
        "| Story | Folder | Flash chars | Flash cites | Flash sources | Pro chars | Pro cites | Pro sources |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        fm = row["flash_metrics"]
        pm = row["pro_metrics"]
        title = row["title"].replace("|", "/")
        table_lines.append(
            f"| {title} | {row['folder']} | {fm['summary_chars']} | {fm['citations_total']} | {fm['sources_total']} | "
            f"{pm['summary_chars']} | {pm['citations_total']} | {pm['sources_total']} |"
        )
    (base_dir / "comparison.md").write_text("\n".join(table_lines), encoding="utf-8")

    return {
        "run_dir": str(base_dir),
        "stories_total": len(stories),
        "stories_selected": len(story_contexts),
        "analyzed": len(results),
        "models": {"flash": flash_model, "pro": pro_model},
        "duration_seconds": round(time.time() - start, 2),
    }


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
