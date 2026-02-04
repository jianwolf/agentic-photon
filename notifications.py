"""Notification system for important stories.

This module handles all output for analyzed stories:
- Markdown reports saved to disk
- Webhook POST notifications
- JSONL alerts file

All notification methods are async and fail gracefully (errors are logged
but don't affect other notifications or the pipeline).

Output Formats:
    Markdown: Full report with metadata and analysis
    Webhook: JSON payload for integration with external systems
    JSONL: One JSON object per line for log aggregation
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import aiohttp

from config import Config
from models.story import Story
from models.research import Analysis

logger = logging.getLogger(__name__)


def _sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize title for use as filename.

    Removes characters that are invalid in filenames and truncates
    to a reasonable length while preserving word boundaries.

    Args:
        title: Story title to sanitize
        max_length: Maximum filename length (before extension)

    Returns:
        Safe filename string
    """
    # Remove invalid filename characters
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', " ", title)
    s = re.sub(r"\s+", " ", s).strip()

    # Truncate at word boundary if too long
    if len(s) > max_length:
        s = s[:max_length]
        last_space = s.rfind(" ")
        if last_space > max_length // 2:
            s = s[:last_space]

    return s.strip()


def _route_tag(route: str | None) -> str | None:
    """Normalize a route label for filenames."""
    if not route:
        return None
    tag = re.sub(r"[^a-z0-9]+", "_", route.lower()).strip("_")
    return tag or None


def _build_report_filename(title: str, timestamp: str, route: str | None) -> str:
    """Build a report filename with an optional route tag."""
    safe_title = _sanitize_filename(title)
    tag = _route_tag(route)
    if tag:
        return f"{timestamp}_{tag}_{safe_title}.md"
    return f"{timestamp}_{safe_title}.md"


def _build_digest_filename(timestamp: str, route: str | None) -> str:
    """Build a digest filename with an optional route tag."""
    tag = _route_tag(route)
    if tag:
        return f"{timestamp}_{tag}_digest.md"
    return f"{timestamp}_digest.md"


async def save_markdown_report(
    story: Story,
    analysis: Analysis,
    reports_dir: Path,
    route: str | None = None,
) -> Path | None:
    """Save markdown report for an analyzed story."""
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = _build_report_filename(story.title, timestamp, route)
        filepath = reports_dir / filename

        pub_date_str = story.pub_date.strftime('%Y-%m-%d %H:%M')
        analyzed_str = now.strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            f"# {story.title}",
            "",
            f"**Published:** {pub_date_str}",
            f"**Source:** {story.source_url}",
            f"**Analyzed:** {analyzed_str}",
            "",
            "---",
            "",
            "## Summary",
            "",
            analysis.summary,
        ]

        if analysis.thought:
            lines.extend([
                "",
                "---",
                "",
                "## Analysis Notes",
                "",
                analysis.thought,
            ])

        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report saved | file=%s", filepath.name)
        return filepath

    except Exception as e:
        logger.error("Report save failed: %s", e, exc_info=True)
        return None


async def save_digest_report(
    digest_markdown: str,
    reports_dir: Path,
    route: str | None = None,
) -> Path | None:
    """Save a per-run digest markdown file."""
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = _build_digest_filename(timestamp, route)
        filepath = reports_dir / filename
        filepath.write_text(digest_markdown, encoding="utf-8")
        logger.info("Digest saved | file=%s", filepath.name)
        return filepath
    except Exception as e:
        logger.error("Digest save failed: %s", e, exc_info=True)
        return None


async def send_webhook(story: Story, analysis: Analysis, url: str) -> bool:
    """Send notification via webhook POST."""
    if not url:
        return True

    payload = {
        "type": "important_story",
        "timestamp": datetime.now().isoformat(),
        "hash": story.hash,
        "title": story.title,
        "pub_date": story.pub_date.isoformat(),
        "source_url": story.source_url,
        "summary": analysis.summary,
        "thought": analysis.thought,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status < 300:
                    logger.debug("Webhook sent | title=%s", story.title[:40])
                    return True
                logger.warning("Webhook failed | status=%d title=%s", resp.status, story.title[:40])
                return False
    except asyncio.TimeoutError:
        logger.warning("Webhook timeout | url=%s title=%s", url[:50], story.title[:40])
        return False
    except Exception as e:
        logger.error("Webhook error: %s (%s)", e, type(e).__name__, exc_info=True)
        return False


async def append_alerts_file(story: Story, analysis: Analysis, filepath: str) -> bool:
    """Append alert to JSONL file."""
    if not filepath:
        return True

    alert = {
        "timestamp": datetime.now().isoformat(),
        "hash": story.hash,
        "title": story.title,
        "source_url": story.source_url,
        "summary": analysis.summary,
        "thought": analysis.thought,
    }

    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Warn if file is getting large (> 100MB)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                logger.warning("Alerts file large | size=%.1fMB path=%s", size_mb, filepath)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logger.error("Alerts file error: %s (%s)", e, type(e).__name__, exc_info=True)
        return False


async def notify(
    story: Story,
    analysis: Analysis,
    config: Config,
    route: str | None = None,
) -> tuple[bool, Path | None]:
    """Send all configured notifications for a story."""
    # Always save report
    report_path = await save_markdown_report(
        story,
        analysis,
        config.reports_dir,
        route=route,
    )
    report_ok = report_path is not None

    # Optional notifications
    webhook_ok = await send_webhook(story, analysis, config.webhook_url)
    alerts_ok = await append_alerts_file(story, analysis, config.alerts_file)

    return report_ok and webhook_ok and alerts_ok, report_path


async def notify_batch(
    items: list[tuple[Story, Analysis]],
    config: Config,
    route: str | None = None,
) -> tuple[int, int, list[Path]]:
    """Send notifications for multiple stories.

    Returns:
        (successful, failed, report_paths) counts and saved report paths
    """
    tasks = [notify(story, analysis, config, route=route) for story, analysis in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = 0
    fail = 0
    report_paths: list[Path] = []

    for result in results:
        if isinstance(result, Exception):
            fail += 1
            continue
        ok_flag, path = result
        if ok_flag:
            ok += 1
        else:
            fail += 1
        if path:
            report_paths.append(path)

    return ok, fail, report_paths
