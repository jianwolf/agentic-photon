#!/usr/bin/env python3
"""Benchmark classifier latency using labeled JSONL inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agents.classifier import ClassifierAgent
from config import Config
from mlx_server import MLXServerManager
from models.story import Story


@dataclass
class Example:
    example_id: str
    story: Story


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _parse_pub_date(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _build_examples(records: Iterable[dict]) -> list[Example]:
    examples: list[Example] = []
    for record in records:
        example_id = record.get("id") or record.get("hash")
        title = record.get("title") or ""
        if not example_id or not title:
            continue

        story = Story(
            title=title,
            description=record.get("description") or "",
            pub_date=_parse_pub_date(record.get("pub_date")),
            source_url=record.get("source_url") or "",
            article_url=record.get("article_url") or "",
        )
        examples.append(Example(example_id=str(example_id), story=story))
    return examples


async def _run_benchmark(examples: list[Example], agent: ClassifierAgent, concurrency: int) -> list[float]:
    semaphore = asyncio.Semaphore(concurrency)
    durations: list[float] = []

    async def classify_one(example: Example) -> None:
        async with semaphore:
            start = time.perf_counter()
            await agent.classify(example.story)
            durations.append(time.perf_counter() - start)

    await asyncio.gather(*(classify_one(ex) for ex in examples))
    return durations


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark classifier latency.")
    parser.add_argument("--labels", required=True, help="JSONL file with story fields.")
    parser.add_argument("--max-items", type=int, default=50, help="Max examples to benchmark.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests.")
    parser.add_argument("--lang", choices=["zh", "en"], help="Prompt language.")
    parser.add_argument("--model", help="Override classifier model (remote).")
    parser.add_argument("--mlx-model", help="Start local MLX server with this model.")
    parser.add_argument("--mlx-port", type=int, default=8080, help="MLX server port.")
    parser.add_argument("--out", help="Write benchmark metrics JSON.")
    args = parser.parse_args()

    records = _load_jsonl(Path(args.labels))
    examples = _build_examples(records)
    if args.max_items > 0:
        examples = examples[: args.max_items]
    if not examples:
        raise SystemExit("No examples found for benchmarking.")

    config = Config.load()
    if args.lang:
        config.language = args.lang
    if args.model:
        config.classifier_model = args.model

    mlx_server: MLXServerManager | None = None
    if args.mlx_model:
        mlx_server = MLXServerManager(model=args.mlx_model, port=args.mlx_port)
        mlx_server.start()
        config.classifier_model = f"openai:{args.mlx_model}@http://127.0.0.1:{args.mlx_port}/v1"

    try:
        agent = ClassifierAgent(config)
        start = time.perf_counter()
        durations = asyncio.run(_run_benchmark(examples, agent, args.concurrency))
        total = time.perf_counter() - start
    finally:
        if mlx_server:
            mlx_server.stop()

    durations_ms = [d * 1000 for d in durations]
    metrics = {
        "items": len(durations_ms),
        "concurrency": args.concurrency,
        "total_s": round(total, 4),
        "avg_ms": round(statistics.mean(durations_ms), 2),
        "p50_ms": round(_percentile(durations_ms, 50), 2),
        "p90_ms": round(_percentile(durations_ms, 90), 2),
        "p95_ms": round(_percentile(durations_ms, 95), 2),
    }

    print(json.dumps(metrics, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
