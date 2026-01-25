#!/usr/bin/env python3
"""Deterministically split labeled JSONL into train/dev/test sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser(description="Split labeled JSONL into train/dev/test.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--dev", type=float, default=0.1, help="Dev fraction.")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction.")
    parser.add_argument("--seed", type=int, default=13, help="Hash seed for splitting.")
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Include unlabeled examples in splits.",
    )
    args = parser.parse_args()

    total = args.train + args.dev + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("Split fractions must sum to 1.0")

    items = _load_jsonl(Path(args.input))

    train_items: list[dict] = []
    dev_items: list[dict] = []
    test_items: list[dict] = []

    for item in items:
        if not args.allow_unlabeled and item.get("label") is None:
            continue

        item_id = item.get("id") or item.get("hash")
        if not item_id:
            continue

        frac = _stable_fraction(str(item_id), args.seed)
        if frac < args.train:
            train_items.append(item)
        elif frac < args.train + args.dev:
            dev_items.append(item)
        else:
            test_items.append(item)

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", train_items)
    _write_jsonl(out_dir / "dev.jsonl", dev_items)
    _write_jsonl(out_dir / "test.jsonl", test_items)

    print(
        f"Wrote splits to {out_dir} | train={len(train_items)} dev={len(dev_items)} test={len(test_items)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
