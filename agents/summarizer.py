"""Summarizer agent for per-run digest generation."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from pydantic_ai import Agent, PromptedOutput, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from config import Config
from models.summary import DigestReport

logger = logging.getLogger(__name__)


SUMMARIZER_PROMPTS = {
    "zh": """你是一名新闻编辑，负责为一次运行生成精炼的摘要（digest）。

你将收到同一次运行生成的多个Markdown报告。**必须完整阅读每份报告的全部内容**，然后输出结构化digest。

## 输出要求（必须符合 DigestReport 结构）
- overview：1-2段，总结本次运行整体发生了什么、重要背景与影响。
- story_summaries：每份报告对应一条摘要，**顺序与输入一致**。
  - title：报告标题
  - takeaway：1-2句关键信息或结论
  - source：若报告中有来源信息则填写，否则留空
  - report_file：报告文件名
- themes：3-5条跨故事的主题或趋势
- watchlist：3-5条后续需要关注的信号/问题

## 约束
1. 不要编造来源或事实。
2. 若报告内容为空或信息不足，takeaway 中需明确说明。
3. 保持可读、逻辑清晰；**不必担心摘要过长**。
4. 如果故事很多，仍需**逐条**给出每个故事的摘要。
5. 输出语言与传入的语言设置保持一致。""",
    "en": """You are a news editor creating a concise per-run digest.

You will receive multiple Markdown reports from the same run. **Read every report in full** and produce a structured digest.

## Output requirements (must conform to DigestReport)
- overview: 1-2 paragraphs summarizing the overall run, key context, and impact.
- story_summaries: one entry per report, **in the same order as input**.
  - title: report title
  - takeaway: 1-2 sentence key takeaway
  - source: source/outlet if present in the report, otherwise empty
  - report_file: report filename
- themes: 3-5 cross-cutting themes or trends
- watchlist: 3-5 follow-up signals/questions to monitor

## Constraints
1. Do not invent sources or facts.
2. If a report is empty or lacks detail, note that in the takeaway.
3. Keep the writing readable and coherent; **length is not a concern**.
4. If there are many stories, still provide a **summary for each one**.
5. Match the output language to the provided language setting.""",
}


@dataclass
class SummarizerContext:
    """Runtime context passed to the summarizer agent.

    Attributes:
        language: Output language ('zh' or 'en')
    """

    language: str = "en"


@dataclass
class ReportDocument:
    """Report content passed to the summarizer prompt."""

    filename: str
    content: str


def _parse_local_model(model_str: str) -> tuple[str, str] | None:
    """Parse local model string into (model_name, base_url) or None if not local."""
    if model_str.startswith("openai:") and "@" in model_str:
        rest = model_str[7:]
        model_name, base_url = rest.split("@", 1)
        return model_name, base_url
    return None


def _create_model(model_str: str):
    """Create a PydanticAI model instance or pass through remote model string."""
    parsed = _parse_local_model(model_str)
    if parsed:
        model_name, base_url = parsed
        logger.info("Using local MLX model | model=%s base_url=%s", model_name, base_url)
        client = AsyncOpenAI(base_url=base_url, api_key="local-model")
        profile = OpenAIModelProfile(supports_json_object_output=False)
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(openai_client=client),
            profile=profile,
        )
    return model_str


def _create_agent(model: str) -> Agent[SummarizerContext, DigestReport]:
    """Create the underlying PydanticAI agent for digest summarization."""
    model_instance = _create_model(model)
    is_local = _parse_local_model(model) is not None
    output_type = PromptedOutput(DigestReport) if is_local else DigestReport

    agent = Agent(
        model_instance,
        output_type=output_type,
        system_prompt=SUMMARIZER_PROMPTS["en"],  # Default fallback
        retries=2,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[SummarizerContext]) -> str:
        return SUMMARIZER_PROMPTS.get(ctx.deps.language, SUMMARIZER_PROMPTS["en"])

    return agent


def _build_user_message(reports: list[ReportDocument]) -> str:
    """Build the user message containing full report markdowns."""
    lines = [
        f"Reports provided: {len(reports)}",
        "Read all reports fully and produce the digest.",
    ]
    for i, report in enumerate(reports, start=1):
        lines.extend(
            [
                "",
                f"=== BEGIN REPORT {i}: {report.filename} ===",
                report.content,
                f"=== END REPORT {i} ===",
            ]
        )
    return "\n".join(lines)


def render_digest_markdown(
    digest: DigestReport,
    report_paths: list[Path],
    label: str | None = None,
) -> str:
    """Render a DigestReport into a human-readable markdown file."""
    generated_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "Run Digest"
    if label:
        title = f"Run Digest ({label})"
    lines = [
        f"# {title}",
        "",
        f"**Generated:** {generated_str}",
        f"**Reports:** {len(report_paths)}",
    ]

    if digest.overview:
        lines.extend(["", "## Overview", "", digest.overview])

    if digest.story_summaries:
        lines.extend(["", "## Story Summaries", ""])
        for story in digest.story_summaries:
            title = story.title or "Untitled"
            takeaway = story.takeaway or ""
            line = f"- **{title}**"
            if takeaway:
                line += f" — {takeaway}"
            if story.source:
                line += f" (Source: {story.source})"
            if story.report_file:
                line += f" (Report: {story.report_file})"
            lines.append(line)

    if digest.themes:
        lines.extend(["", "## Themes", ""])
        lines.extend([f"- {theme}" for theme in digest.themes])

    if digest.watchlist:
        lines.extend(["", "## Watchlist", ""])
        lines.extend([f"- {item}" for item in digest.watchlist])

    if report_paths:
        lines.extend(["", "## Included Reports", ""])
        lines.extend([f"- {path.name}" for path in report_paths])

    return "\n".join(lines)


class SummarizerAgent:
    """Generates a per-run digest from full markdown reports."""

    def __init__(self, config: Config):
        """Initialize the summarizer agent.

        Args:
            config: Application configuration with model and language settings
        """
        self.config = config
        self._agent = _create_agent(config.summary_model)
        self._context = SummarizerContext(language=config.language)

    async def summarize_paths(
        self,
        report_paths: list[Path],
    ) -> tuple[DigestReport, int, int]:
        """Generate a digest from the provided report paths.

        Args:
            report_paths: List of markdown report file paths (same run)

        Returns:
            Tuple of (digest report, input_tokens, output_tokens)
        """
        reports: list[ReportDocument] = []
        for path in report_paths:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error("Digest read failed | file=%s error=%s", path.name, e, exc_info=True)
                content = f"ERROR: unable to read report ({type(e).__name__}: {e})"
            reports.append(ReportDocument(filename=path.name, content=content))

        message = _build_user_message(reports)
        result = await self._agent.run(
            message,
            deps=self._context,
            usage_limits=UsageLimits(request_limit=3),
        )
        usage = result.usage()
        input_tokens = usage.request_tokens or 0
        output_tokens = usage.response_tokens or 0
        logger.info(
            "Digest generated | reports=%d input_tokens=%d output_tokens=%d",
            len(report_paths),
            input_tokens,
            output_tokens,
        )
        return result.output, input_tokens, output_tokens
