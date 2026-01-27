"""LLM-as-a-judge for researcher report quality.

This script scores flash vs pro reports on depth and grounding using an LLM judge.
It reads flash.json/pro.json pairs produced by `python main.py compare`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+")
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class ReportScore(BaseModel):
    """Score for a single report."""

    depth: float = Field(description="Depth score from 0-5 (can use .5 increments)")
    grounding: float = Field(description="Grounding score from 0-5 (can use .5 increments)")
    strengths: list[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Key weaknesses")


class PairVerdict(BaseModel):
    """Verdict for a flash vs pro pair."""

    winner: Literal["flash", "pro", "tie"] = Field(description="Overall winner")
    rationale: str = Field(description="Brief rationale (2-4 sentences, no chain-of-thought)")
    flash: ReportScore
    pro: ReportScore


@dataclass
class JudgeContext:
    """Context for judge agent."""

    language: str = "en"


def _count_urls(text: str) -> int:
    return len(_URL_PATTERN.findall(text or ""))


def _count_citations(text: str) -> int:
    return len(_CITATION_PATTERN.findall(text or ""))


def _report_stats(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", "") or ""
    thought = report.get("thought", "") or ""
    key_points = report.get("key_points", []) or []
    related_topics = report.get("related_topics", []) or []

    citations = _count_citations(summary + "\n" + "\n".join(key_points))
    urls = _count_urls(thought)
    flags = []
    if not summary.strip():
        flags.append("empty_summary")
    if "Analysis error" in thought:
        flags.append("analysis_error")
    if "模拟" in thought or "simulated" in thought.lower():
        flags.append("simulated_sources")

    return {
        "summary_chars": len(summary),
        "key_points_count": len(key_points),
        "related_topics_count": len(related_topics),
        "citations_count": citations,
        "urls_count": urls,
        "flags": flags,
    }


def _build_prompt(
    story: dict[str, Any],
    flash_report: dict[str, Any],
    pro_report: dict[str, Any],
    flash_stats: dict[str, Any],
    pro_stats: dict[str, Any],
) -> str:
    return f"""
You are an impartial judge scoring two reports about the same story.
Only use the provided text; do NOT assume external facts.

Scoring rubric (0-5, allow .5):
- Depth: specificity, causal reasoning, technical detail, and structured analysis.
  *0 = empty, 1 = surface, 3 = solid, 5 = deep and well-structured.*
- Grounding: evidence-based claims, clear source URLs, and minimal speculation.
  *0 = no grounding, 1 = citations but no URLs or unsupported details,
   3 = some URLs tied to claims, 5 = strong URL-backed claims with clear traceability.*

Penalties:
- Missing or placeholder sources/URLs => reduce grounding.
- “Simulated” or hypothetical sources => reduce grounding.
- Empty summary or error output => depth/grounding should be 0.

Return a JSON object that matches the schema: PairVerdict.
Keep rationale brief (2-4 sentences), no chain-of-thought.

Story metadata:
- Title: {story.get('title', '')}
- Source URL: {story.get('source_url', '')}
- Article URL: {story.get('article_url', '')}
- Published: {story.get('pub_date', '')}

Flash report:
Summary:
{flash_report.get('summary', '')}

Thought:
{flash_report.get('thought', '')}

Key points:
{flash_report.get('key_points', [])}

Related topics:
{flash_report.get('related_topics', [])}

Flash stats:
{json.dumps(flash_stats, ensure_ascii=False)}

Pro report:
Summary:
{pro_report.get('summary', '')}

Thought:
{pro_report.get('thought', '')}

Key points:
{pro_report.get('key_points', [])}

Related topics:
{pro_report.get('related_topics', [])}

Pro stats:
{json.dumps(pro_stats, ensure_ascii=False)}
""".strip()


def _create_agent(model: str) -> Agent[JudgeContext, PairVerdict]:
    system_prompt = (
        "You are a strict evaluation judge for report quality. "
        "Be consistent, conservative, and avoid hallucinating facts."
    )
    agent = Agent(
        model,
        output_type=PairVerdict,
        system_prompt=system_prompt,
        retries=2,
    )

    @agent.system_prompt
    def _dynamic_prompt(ctx: RunContext[JudgeContext]) -> str:  # noqa: ARG001
        return system_prompt

    return agent


def _combined_score(depth: float, grounding: float) -> float:
    return round(0.6 * depth + 0.4 * grounding, 2)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_pairs(input_dir: Path) -> list[dict[str, Path]]:
    pairs: list[dict[str, Path]] = []
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        flash = subdir / "flash.json"
        pro = subdir / "pro.json"
        if flash.exists() and pro.exists():
            pairs.append({"dir": subdir, "flash": flash, "pro": pro})
    return pairs


async def _judge_pair(
    agent: Agent[JudgeContext, PairVerdict],
    ctx: JudgeContext,
    pair: dict[str, Path],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        flash_data = json.loads(pair["flash"].read_text(encoding="utf-8"))
        pro_data = json.loads(pair["pro"].read_text(encoding="utf-8"))

        story = flash_data.get("story", {})
        flash_report = flash_data.get("report", {})
        pro_report = pro_data.get("report", {})

        flash_stats = _report_stats(flash_report)
        pro_stats = _report_stats(pro_report)

        prompt = _build_prompt(story, flash_report, pro_report, flash_stats, pro_stats)

        try:
            result = await agent.run(prompt, deps=ctx)
            verdict = result.output
        except Exception as exc:  # noqa: BLE001
            logger.error("Judge failed for %s: %s", pair["dir"].name, exc, exc_info=True)
            verdict = PairVerdict(
                winner="tie",
                rationale=f"Judge error: {type(exc).__name__}",
                flash=ReportScore(depth=0, grounding=0, strengths=[], weaknesses=["judge_error"]),
                pro=ReportScore(depth=0, grounding=0, strengths=[], weaknesses=["judge_error"]),
            )

        payload = {
            "story": {
                "title": story.get("title", ""),
                "source_url": story.get("source_url", ""),
                "article_url": story.get("article_url", ""),
                "pub_date": story.get("pub_date", ""),
            },
            "flash": {
                "metrics": flash_stats,
                "score": verdict.flash.model_dump(),
                "combined": _combined_score(verdict.flash.depth, verdict.flash.grounding),
            },
            "pro": {
                "metrics": pro_stats,
                "score": verdict.pro.model_dump(),
                "combined": _combined_score(verdict.pro.depth, verdict.pro.grounding),
            },
            "winner": verdict.winner,
            "rationale": verdict.rationale,
        }
        return payload


async def run_judge(
    input_dir: Path,
    model: str,
    output_json: Path,
    output_md: Path,
    max_concurrent: int = 2,
    limit: int | None = None,
) -> dict[str, Any]:
    agent = _create_agent(model)
    ctx = JudgeContext()

    pairs = _collect_pairs(input_dir)
    if limit:
        pairs = pairs[:limit]

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.create_task(_judge_pair(agent, ctx, pair, semaphore))
        for pair in pairs
    ]

    results: list[dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    # Summary stats
    flash_scores = [r["flash"]["combined"] for r in results]
    pro_scores = [r["pro"]["combined"] for r in results]
    winners = {
        "flash": sum(1 for r in results if r["winner"] == "flash"),
        "pro": sum(1 for r in results if r["winner"] == "pro"),
        "tie": sum(1 for r in results if r["winner"] == "tie"),
    }

    summary = {
        "input_dir": str(input_dir),
        "model": model,
        "stories": len(results),
        "averages": {
            "flash": round(sum(flash_scores) / len(flash_scores), 2) if flash_scores else 0,
            "pro": round(sum(pro_scores) / len(pro_scores), 2) if pro_scores else 0,
        },
        "winners": winners,
        "results": results,
    }

    _write_json(output_json, summary)

    md_lines = [
        "# LLM Judge Report",
        "",
        f"- Input: {input_dir}",
        f"- Judge model: {model}",
        f"- Stories: {len(results)}",
        f"- Winners: flash={winners['flash']} pro={winners['pro']} tie={winners['tie']}",
        "",
        "| Story | Winner | Flash (D/G/C) | Pro (D/G/C) | Rationale |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        title = r["story"]["title"].replace("|", "/")
        flash_score = r["flash"]["score"]
        pro_score = r["pro"]["score"]
        md_lines.append(
            f"| {title} | {r['winner']} | "
            f"{flash_score['depth']}/{flash_score['grounding']}/{r['flash']['combined']} | "
            f"{pro_score['depth']}/{pro_score['grounding']}/{r['pro']['combined']} | "
            f"{r['rationale']} |"
        )

    _write_markdown(output_md, md_lines)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge for report quality")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Compare run directory (e.g., reports/compare/20260126_214722)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google-gla:gemini-3-pro-preview",
        help="Judge model (PydanticAI model string)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: eval/judge_<timestamp>.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown path (default: eval/judge_<timestamp>.md)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Max concurrent judge calls (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of stories (0 = all)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or Path("eval") / f"judge_{stamp}.json"
    out_md = args.out_md or Path("eval") / f"judge_{stamp}.md"

    out_json.parent.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit > 0 else None

    summary = asyncio.run(
        run_judge(
            input_dir=input_dir,
            model=args.model,
            output_json=out_json,
            output_md=out_md,
            max_concurrent=args.max_concurrent,
            limit=limit,
        )
    )
    print(json.dumps({k: summary[k] for k in ("input_dir", "model", "stories", "averages", "winners")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
