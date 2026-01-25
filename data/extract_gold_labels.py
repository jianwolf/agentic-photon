#!/usr/bin/env python3
"""Extract manual-only labels from a mixed seed set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract manual labels from seed JSONL.")
    parser.add_argument("--input", default="data/labels/seed.jsonl", help="Input JSONL file.")
    parser.add_argument("--out", default="data/labels/gold.jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--notes-prefix",
        default="manual",
        help="Notes prefix used to mark manual labels.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = args.notes_prefix.lower()
    kept = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            notes = (item.get("notes") or "").lower()
            if not notes.startswith(prefix):
                continue
            if item.get("label") is None:
                continue
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} manual labels to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
