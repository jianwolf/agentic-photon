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


async def save_markdown_report(
    story: Story,
    analysis: Analysis,
    reports_dir: Path,
) -> Path | None:
    """Save markdown report for an analyzed story."""
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{_sanitize_filename(story.title)}.md"
        filepath = reports_dir / filename

        lines = [
            f"# {story.title}",
            "",
            f"**Published:** {story.pub_date.strftime('%Y-%m-%d %H:%M')}",
            f"**Source:** {story.source_url}",
            f"**Analyzed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        logger.info(f"Report saved: {filepath.name}")
        return filepath

    except Exception as e:
        logger.error(f"Report save failed: {e}")
        return None


async def send_webhook(story: Story, analysis: Analysis, url: str) -> bool:
    """Send notification via webhook POST."""
    if not url:
        return True

    payload = {
        "type": "important_story",
        "timestamp": datetime.now().isoformat(),
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
                    logger.debug(f"Webhook sent: {story.title[:40]}")
                    return True
                logger.warning(f"Webhook {resp.status}: {story.title[:40]}")
                return False
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return False


async def append_alerts_file(story: Story, analysis: Analysis, filepath: str) -> bool:
    """Append alert to JSONL file."""
    if not filepath:
        return True

    alert = {
        "timestamp": datetime.now().isoformat(),
        "title": story.title,
        "source_url": story.source_url,
        "summary": analysis.summary,
        "thought": analysis.thought,
    }

    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logger.error(f"Alerts file error: {e}")
        return False


async def notify(story: Story, analysis: Analysis, config: Config) -> bool:
    """Send all configured notifications for a story."""
    # Always save report
    report_ok = await save_markdown_report(story, analysis, config.reports_dir) is not None

    # Optional notifications
    webhook_ok = await send_webhook(story, analysis, config.webhook_url)
    alerts_ok = await append_alerts_file(story, analysis, config.alerts_file)

    return report_ok and webhook_ok and alerts_ok


async def notify_batch(
    items: list[tuple[Story, Analysis]],
    config: Config,
) -> tuple[int, int]:
    """Send notifications for multiple stories.

    Returns:
        (successful, failed) counts
    """
    tasks = [notify(story, analysis, config) for story, analysis in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = sum(1 for r in results if r is True)
    return ok, len(results) - ok
