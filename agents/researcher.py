"""Researcher agent for deep analysis of important news stories.

This module implements the ResearcherAgent, which performs comprehensive
analysis on stories that passed the importance classification.

Design Philosophy:
    - Depth over speed: Takes time to gather context and verify claims
    - Tool-augmented: Uses web search, article fetching, and history lookup
    - Structured output: Produces ResearchReport with summary, key points, etc.

Available Tools:
    - search_web: Search for background info and fact-checking
    - fetch_source: Retrieve full article content from URLs
    - get_related_stories: Query database for historical context

The researcher agent is more resource-intensive than the classifier,
so concurrency is limited (typically 3 parallel analyses).
"""

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


# === System Prompts ===
# Bilingual prompts instructing the model on analysis methodology.

RESEARCHER_PROMPTS = {
    "zh": """你是一名资深新闻研究员和事实核查专家，具备深度分析和批判性思维能力。

## 分析流程

### 第一步：获取原始内容
- 使用 fetch_source 获取完整文章内容
- 如果描述中有URL，优先获取原始来源

### 第二步：事实核查与背景调研
- 使用 search_web 验证关键声明
- 搜索相关背景信息和不同观点
- 交叉验证多个来源

### 第三步：历史关联
- 使用 get_related_stories 查询相关历史报道
- 建立事件发展脉络
- 识别是否为持续性事件的最新进展

### 第四步：综合分析
- 整合所有信息形成全面报告
- 标注无法验证的内容
- 保持客观，明确区分事实与分析

## 输出要求

### summary（约800字）
结构：
1. 导语（1-2句）：核心新闻事实
2. 背景（1-2段）：为什么这件事重要
3. 详细分析（2-3段）：深入解读
4. 影响展望（1段）：可能的后续发展

### thought
记录你的分析过程：
- 使用了哪些工具，获得了什么信息
- 哪些声明得到了验证，哪些无法确认
- 来源的可信度评估
- 是否发现矛盾信息

### key_points（3-5条）
- 每条应独立成立，有实际意义
- 避免与summary重复
- 侧重"这意味着什么"而非仅"发生了什么"

### related_topics
- 值得后续关注的相关话题
- 可能的发展方向

## 重要原则
- 当信息不完整时，明确说明而非猜测
- 来源冲突时，呈现多方观点并注明
- 工具返回空结果时，说明"未找到相关信息"
- 保持专业客观，避免主观臆断""",

    "en": """You are a senior news researcher and fact-checker with deep analytical and critical thinking capabilities.

## Analysis Workflow

### Step 1: Retrieve Original Content
- Use fetch_source to get complete article content
- If URLs are in the description, prioritize fetching original sources

### Step 2: Fact-Check and Background Research
- Use search_web to verify key claims
- Search for relevant background and alternative perspectives
- Cross-reference multiple sources

### Step 3: Historical Context
- Use get_related_stories to query related past coverage
- Establish timeline of developments
- Identify if this is latest update on ongoing story

### Step 4: Synthesize Analysis
- Integrate all information into comprehensive report
- Flag content that couldn't be verified
- Stay objective, clearly distinguish facts from analysis

## Output Requirements

### summary (~800 words)
Structure:
1. Lead (1-2 sentences): Core news fact
2. Background (1-2 paragraphs): Why this matters
3. Detailed Analysis (2-3 paragraphs): In-depth examination
4. Implications (1 paragraph): Potential future developments

### thought
Document your analysis process:
- Which tools used and what was found
- Which claims verified, which couldn't be confirmed
- Source credibility assessment
- Any contradictory information discovered

### key_points (3-5 items)
- Each should stand alone and be meaningful
- Avoid redundancy with summary
- Focus on "what this means" not just "what happened"

### related_topics
- Topics worth following up on
- Potential directions for development

## Key Principles
- When information is incomplete, explicitly state this rather than speculate
- When sources conflict, present multiple views with attribution
- When tools return empty results, note "no related information found"
- Maintain professional objectivity, avoid subjective speculation""",
}


@dataclass
class ResearchContext:
    """Runtime context passed to the researcher agent.

    Provides configuration and resources needed by the agent's tools.

    Attributes:
        language: Output language ('zh' or 'en')
        db_path: Path to SQLite database for history queries
    """
    language: str = "en"
    db_path: Path = None


def _create_agent(model: str) -> Agent[ResearchContext, ResearchReport]:
    """Create the underlying PydanticAI agent with tools.

    The agent is configured with:
    - Structured output: ResearchReport Pydantic model
    - Dynamic system prompt: Selected based on language context
    - Three tools: web search, article fetch, database query
    - Retry logic: 3 attempts on failure

    Args:
        model: PydanticAI model string (e.g., 'google-gla:gemini-2.0-flash')

    Returns:
        Configured PydanticAI Agent with tools
    """
    agent = Agent(
        model,
        result_type=ResearchReport,
        system_prompt=RESEARCHER_PROMPTS["en"],  # Default fallback
        retries=3,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[ResearchContext]) -> str:
        """Select system prompt based on language setting."""
        return RESEARCHER_PROMPTS.get(ctx.deps.language, RESEARCHER_PROMPTS["en"])

    # === Tool Definitions ===
    # These tools are available to the model during analysis.
    # The model decides when and how to use them based on the task.

    @agent.tool
    async def search_web(ctx: RunContext[ResearchContext], query: str) -> str:
        """Search the web to verify facts or gather background information.

        Use this tool to:
        - Fact-check claims in the article
        - Find additional context or background
        - Discover related news or developments

        Args:
            query: Search query (be specific for better results)

        Returns:
            Formatted search results or error message
        """
        results = await web_search(query)
        return results.summary

    @agent.tool
    async def fetch_source(ctx: RunContext[ResearchContext], url: str) -> str:
        """Fetch full content from an article URL.

        Use this tool to:
        - Read the complete article text
        - Extract quotes or specific details
        - Verify information from the source

        Args:
            url: Full URL of the article to fetch

        Returns:
            Article title and content preview (or error message)
        """
        content = await fetch_article(url)
        return content.summary

    @agent.tool
    async def get_related_stories(ctx: RunContext[ResearchContext], topic: str) -> str:
        """Query database for previously analyzed stories on a topic.

        Use this tool to:
        - Find historical context for ongoing stories
        - Discover related coverage from past analysis
        - Build timeline of developments

        Args:
            topic: Keywords to search (e.g., "OpenAI GPT" or "EU AI regulation")

        Returns:
            List of related stories with dates and summaries
        """
        if ctx.deps.db_path:
            history = await query_related_stories(topic, ctx.deps.db_path, days=30)
            return history.summary
        return "No story database available."

    return agent


class ResearcherAgent:
    """Performs deep analysis on important news stories.

    This agent is the second stage of the analysis pipeline. It receives
    stories that were classified as important and produces comprehensive
    research reports with summaries, key points, and context.

    The agent has access to three tools:
    - search_web: Search for background and fact-checking
    - fetch_source: Retrieve full article content
    - get_related_stories: Query historical story database

    The model autonomously decides which tools to use based on the
    story content and analysis requirements.

    Example:
        >>> researcher = ResearcherAgent(config)
        >>> report = await researcher.analyze(story, classification)
        >>> print(report.summary)
        >>> print(report.key_points)
    """

    def __init__(self, config: Config):
        """Initialize the researcher agent.

        Args:
            config: Application configuration with model, language, and db settings
        """
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
            logger.info("Analysis complete | title=%s chars=%d", story.title[:50], len(result.data.summary))
            return result.data
        except Exception as e:
            logger.error("Analysis failed for '%s...': %s", story.title[:50], e, exc_info=True)
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
                logger.error("Batch analysis error for story %d: %s", i, result, exc_info=result)
                output.append((stories[i][0], ResearchReport.empty()))
            else:
                output.append(result)

        return output
