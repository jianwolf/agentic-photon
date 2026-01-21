"""Researcher agent for deep analysis of important news stories."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext

from config import Config
from models.story import Story
from models.classification import ClassificationResult
from models.research import ResearchReport
from tools.search import web_search
from tools.fetch import fetch_article
from tools.database import query_related_stories

logger = logging.getLogger(__name__)

# System prompts by language
RESEARCHER_PROMPTS = {
    "zh": """你是一名资深新闻研究员和事实核查专家。

任务：
1. 对重要新闻进行全面分析
2. 使用工具验证关键声明
3. 提供相关背景信息
4. 分析潜在影响

输出要求（中文）：
- summary：完整摘要，约1000字
- thought：来源分析和核实记录
- key_points：3-5个关键要点
- related_topics：相关话题

保持客观准确，避免推测。""",

    "en": """You are a senior news researcher and fact-checker.

Tasks:
1. Conduct comprehensive analysis of important news
2. Use tools to verify key claims
3. Provide relevant background context
4. Analyze potential implications

Output requirements:
- summary: Complete summary, ~600 words
- thought: Source analysis and verification notes
- key_points: 3-5 key takeaways
- related_topics: Related topics for follow-up

Remain objective and accurate. Avoid speculation.""",
}


@dataclass
class ResearchContext:
    """Context passed to the researcher agent."""
    language: str = "en"
    db_path: Path = None


def _create_agent(model: str) -> Agent[ResearchContext, ResearchReport]:
    """Create the underlying PydanticAI agent with tools."""
    agent = Agent(
        model,
        result_type=ResearchReport,
        system_prompt=RESEARCHER_PROMPTS["en"],
        retries=3,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[ResearchContext]) -> str:
        return RESEARCHER_PROMPTS.get(ctx.deps.language, RESEARCHER_PROMPTS["en"])

    @agent.tool
    async def search_web(ctx: RunContext[ResearchContext], query: str) -> str:
        """Search the web to verify facts or gather background.

        Args:
            query: Search query
        """
        results = await web_search(query)
        return results.summary

    @agent.tool
    async def fetch_source(ctx: RunContext[ResearchContext], url: str) -> str:
        """Fetch full content from an article URL.

        Args:
            url: Article URL
        """
        content = await fetch_article(url)
        return content.summary

    @agent.tool
    async def get_related_stories(ctx: RunContext[ResearchContext], topic: str) -> str:
        """Query database for previously analyzed stories on a topic.

        Args:
            topic: Topic keywords to search
        """
        if ctx.deps.db_path:
            history = await query_related_stories(topic, ctx.deps.db_path, days=30)
            return history.summary
        return "No story database available."

    return agent


class ResearcherAgent:
    """Performs deep analysis on important news stories.

    Uses tools for web search, article fetching, and history lookup
    to produce comprehensive research reports.
    """

    def __init__(self, config: Config):
        self.config = config
        self._agent = _create_agent(config.researcher_model)
        self._context = ResearchContext(
            language=config.language,
            db_path=config.db_path,
        )

    async def analyze(
        self,
        story: Story,
        classification: ClassificationResult | None = None,
    ) -> ResearchReport:
        """Analyze a story in depth.

        Args:
            story: Story to analyze
            classification: Optional classification for context

        Returns:
            Research report with summary and analysis
        """
        publishers = ", ".join(story.publishers) if story.publishers else "Unknown"

        # Build context from classification
        context_lines = []
        if classification:
            context_lines.append(f"Category: {classification.category.value}")
            if classification.reasoning:
                context_lines.append(f"Classification: {classification.reasoning}")
        context = "\n".join(context_lines)

        message = f"""Analyze this news story:

Title: {story.title}
Publisher: {publishers}
Published: {story.pub_date.strftime("%Y-%m-%d %H:%M")}
{context}

Description:
{story.description[:1000] if story.description else "No description"}

Instructions:
1. Summarize key information comprehensively
2. Use search to verify claims and add background
3. Check for related stories in database
4. Assess significance and impact"""

        try:
            result = await self._agent.run(message, deps=self._context)
            logger.info(f"Analyzed: {story.title[:50]}... ({len(result.data.summary)} chars)")
            return result.data
        except Exception as e:
            logger.error(f"Analysis failed for '{story.title[:50]}...': {e}")
            return ResearchReport.empty()

    async def analyze_batch(
        self,
        stories: list[tuple[Story, ClassificationResult | None]],
        max_concurrent: int = 3,
    ) -> list[tuple[Story, ResearchReport]]:
        """Analyze multiple stories concurrently.

        Args:
            stories: List of (story, classification) tuples
            max_concurrent: Max concurrent API calls

        Returns:
            List of (story, report) tuples
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_one(
            story: Story,
            classification: ClassificationResult | None,
        ) -> tuple[Story, ResearchReport]:
            async with semaphore:
                return story, await self.analyze(story, classification)

        tasks = [analyze_one(s, c) for s, c in stories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis error: {result}")
                output.append((stories[i][0], ResearchReport.empty()))
            else:
                output.append(result)

        return output
