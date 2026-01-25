#!/usr/bin/env python3
"""Collect recent RSS items into a JSONL labeling pool."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from datetime import timezone
from pathlib import Path
from typing import Iterable

from config import Config
from feeds import fetch_all_feeds
from models.story import Story


def _story_to_record(story: Story) -> dict[str, object]:
    return {
        "id": story.hash,
        "title": story.title,
        "description": story.description or "",
        "source_url": story.source_url,
        "article_url": story.article_url or "",
        "pub_date": story.pub_date.astimezone(timezone.utc).isoformat(),
        "label": None,
        "category": "",
        "notes": "",
    }


def _dedupe_stories(stories: Iterable[Story]) -> list[Story]:
    deduped: dict[str, Story] = {}
    for story in stories:
        deduped[story.hash] = story
    return list(deduped.values())


async def _collect_pool(args: argparse.Namespace) -> int:
    config = Config.load()
    stories = await fetch_all_feeds(
        config.rss_urls,
        max_age_hours=args.max_age_hours,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
    )

    stories = _dedupe_stories(stories)

    if args.max_items > 0 and len(stories) > args.max_items:
        rng = random.Random(args.seed)
        rng.shuffle(stories)
        stories = stories[: args.max_items]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for story in stories:
            handle.write(json.dumps(_story_to_record(story), ensure_ascii=False) + "\n")

    return len(stories)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect RSS items for labeling.")
    parser.add_argument("--out", default="data/labels/pool.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-age-hours", type=int, default=168, help="Lookback window.")
    parser.add_argument("--max-items", type=int, default=200, help="Max items (0 = all).")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling.")
    parser.add_argument("--timeout", type=int, default=30, help="Feed request timeout.")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent feeds.")
    args = parser.parse_args()

    count = asyncio.run(_collect_pool(args))
    print(f"Wrote {count} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
