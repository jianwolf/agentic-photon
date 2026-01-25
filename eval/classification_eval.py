#!/usr/bin/env python3
"""Evaluate classifier performance on labeled JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agents.classifier import ClassifierAgent
from config import Config
from mlx_server import MLXServerManager
from models.story import Story


@dataclass
class EvalExample:
    """Normalized evaluation example."""

    example_id: str
    story: Story
    label: bool
    category: str | None = None


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _parse_label(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return None


def _parse_pub_date(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _build_examples(records: Iterable[dict]) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for record in records:
        label = _parse_label(record.get("label"))
        if label is None:
            continue

        example_id = record.get("id") or record.get("hash")
        if not example_id:
            continue

        title = record.get("title") or ""
        if not title:
            continue

        story = Story(
            title=title,
            description=record.get("description") or "",
            pub_date=_parse_pub_date(record.get("pub_date")),
            source_url=record.get("source_url") or "",
            article_url=record.get("article_url") or "",
        )

        examples.append(
            EvalExample(
                example_id=str(example_id),
                story=story,
                label=label,
                category=record.get("category") or None,
            )
        )
    return examples


async def _run_predictions(
    examples: list[EvalExample],
    agent: ClassifierAgent,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def classify_one(example: EvalExample) -> dict:
        async with semaphore:
            result = await agent.classify(example.story)
        return {
            "id": example.example_id,
            "is_important": result.is_important,
            "confidence": result.confidence,
            "category": result.category.value,
            "reasoning": result.reasoning,
        }

    tasks = [classify_one(example) for example in examples]
    for output in await asyncio.gather(*tasks):
        results.append(output)
    return results


def _evaluate(predictions: Iterable[dict], labels: dict[str, EvalExample]) -> dict:
    tp = fp = tn = fn = 0
    matched = 0
    category_total = 0
    category_correct = 0

    for pred in predictions:
        pred_id = pred.get("id") or pred.get("hash")
        if not pred_id or pred_id not in labels:
            continue
        matched += 1
        example = labels[pred_id]
        pred_label = bool(pred.get("is_important"))

        if pred_label and example.label:
            tp += 1
        elif pred_label and not example.label:
            fp += 1
        elif not pred_label and not example.label:
            tn += 1
        else:
            fn += 1

        if example.category and pred.get("category"):
            category_total += 1
            if str(pred.get("category")) == str(example.category):
                category_correct += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    metrics = {
        "total": total,
        "matched": matched,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "label_positive_rate": round((tp + fn) / total, 4) if total else 0.0,
        "prediction_positive_rate": round((tp + fp) / total, 4) if total else 0.0,
    }

    if category_total:
        metrics["category_accuracy"] = round(category_correct / category_total, 4)

    return metrics


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracy.")
    parser.add_argument("--labels", required=True, help="Labeled JSONL file.")
    parser.add_argument("--predictions", help="Predictions JSONL file.")
    parser.add_argument("--out-predictions", help="Write predictions JSONL.")
    parser.add_argument("--out-metrics", help="Write metrics JSON.")
    parser.add_argument("--max-items", type=int, default=0, help="Limit examples (0 = all).")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent requests.")
    parser.add_argument("--lang", choices=["zh", "en"], help="Output language for prompts.")
    parser.add_argument("--model", help="Override classifier model (remote).")
    parser.add_argument("--mlx-model", help="Start local MLX server with this model.")
    parser.add_argument("--mlx-port", type=int, default=8080, help="MLX server port.")
    args = parser.parse_args()

    labels_records = _load_jsonl(Path(args.labels))
    examples = _build_examples(labels_records)
    if args.max_items > 0:
        examples = examples[: args.max_items]

    labels_map = {example.example_id: example for example in examples}
    if not labels_map:
        raise SystemExit("No labeled examples found.")

    predictions: list[dict]
    mlx_server: MLXServerManager | None = None

    try:
        if args.predictions:
            predictions = _load_jsonl(Path(args.predictions))
        else:
            config = Config.load()
            if args.lang:
                config.language = args.lang
            if args.model:
                config.classifier_model = args.model
            if args.mlx_model:
                mlx_server = MLXServerManager(model=args.mlx_model, port=args.mlx_port)
                mlx_server.start()
                config.classifier_model = f"openai:{args.mlx_model}@http://127.0.0.1:{args.mlx_port}/v1"

            agent = ClassifierAgent(config)
            predictions = asyncio.run(
                _run_predictions(examples, agent, concurrency=args.concurrency)
            )

        if args.out_predictions:
            _write_jsonl(Path(args.out_predictions), predictions)
    finally:
        if mlx_server:
            mlx_server.stop()

    metrics = _evaluate(predictions, labels_map)
    print(json.dumps(metrics, indent=2))

    if args.out_metrics:
        Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_metrics).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
