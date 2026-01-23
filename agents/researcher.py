"""Researcher agent for deep analysis of important news stories.

This module implements the ResearcherAgent, which performs comprehensive
analysis on stories that passed the importance classification.

Architecture (v2 - Grounded):
    - Single Gemini API call per story with Google Search grounding
    - All context (article content, related stories) pre-fetched and passed in prompt
    - No custom function tools (incompatible with Google built-in tools)
    - WebSearchTool provides real-time web context via Gemini grounding

Context Flow:
    1. URL Fetch (deterministic): Full article content
    2. Hybrid RAG (deterministic): Related stories from database
    3. Researcher (Gemini + grounding): Single call with all context

This design provides:
    - Cost efficiency: 1 API call per story (vs 1-15 with tools)
    - Real-time context: Google Search grounding for current information
    - Historical context: Pre-fetched related stories from local database
"""

import asyncio
import logging
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, UsageLimits

from config import Config
from models.story import Story
from models.classification import ClassificationResult
from models.research import ResearchReport

logger = logging.getLogger(__name__)


# === System Prompts ===
# Bilingual prompts instructing the model on analysis methodology.
# Context is now provided in the user message, not via tools.

RESEARCHER_PROMPTS = {
    "zh": """你是一名资深新闻研究员和事实核查专家，具备深度分析和批判性思维能力。

## 你将收到的信息

1. **新闻标题和元数据** - 来自RSS订阅源
2. **文章全文** - 已获取的原始内容（如果可用）
3. **相关历史报道** - 来自我们数据库的过往分析

## 分析流程

### 第一步：理解内容
- 仔细阅读提供的文章全文
- 注意关键声明、数据和引用

### 第二步：网络搜索验证
- 使用网络搜索功能验证关键声明
- 搜索相关背景信息和不同观点
- 获取最新的相关发展

### 第三步：历史关联
- 查看提供的相关历史报道
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
- 使用了哪些搜索查询
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
- 搜索无结果时，说明"未找到相关信息"
- 保持专业客观，避免主观臆断""",

    "en": """You are a senior news researcher and fact-checker with deep analytical and critical thinking capabilities.

## Information You Will Receive

1. **News title and metadata** - From RSS feed
2. **Full article content** - Pre-fetched original content (if available)
3. **Related past coverage** - Previous analyses from our database

## Analysis Workflow

### Step 1: Understand Content
- Carefully read the provided article content
- Note key claims, data, and quotes

### Step 2: Web Search Verification
- Use web search to verify key claims
- Search for relevant background and alternative perspectives
- Get latest related developments

### Step 3: Historical Context
- Review the provided related stories
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
- What search queries you used
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
- When search returns nothing relevant, note "no related information found"
- Maintain professional objectivity, avoid subjective speculation""",
}


@dataclass
class ResearchContext:
    """Runtime context passed to the researcher agent.

    Attributes:
        language: Output language ('zh' or 'en')
    """
    language: str = "en"


@dataclass
class StoryContext:
    """Pre-fetched context for a story to be analyzed.

    This replaces the tool-based context gathering. All information
    is gathered before the researcher agent runs.

    Attributes:
        story: The story to analyze
        classification: Classification result (for category context)
        article_content: Full article text from URL fetch (may be empty if fetch failed)
        related_stories: Formatted string of related stories from hybrid RAG
    """
    story: Story
    classification: ClassificationResult | None = None
    article_content: str = ""
    related_stories: str = ""


def _create_agent(model: str) -> Agent[ResearchContext, ResearchReport]:
    """Create the underlying PydanticAI agent with Google Search grounding.

    The agent is configured with:
    - Structured output: ResearchReport Pydantic model
    - Dynamic system prompt: Selected based on language context
    - WebSearchTool: Google Search grounding for real-time information
    - No custom function tools (incompatible with built-in tools on Google)

    Args:
        model: PydanticAI model string (e.g., 'google-gla:gemini-3-flash-preview')

    Returns:
        Configured PydanticAI Agent with grounding
    """
    # Import WebSearchTool for grounding
    from pydantic_ai import WebSearchTool

    agent = Agent(
        model,
        output_type=ResearchReport,
        system_prompt=RESEARCHER_PROMPTS["en"],  # Default fallback
        builtin_tools=[WebSearchTool()],  # Google Search grounding
        retries=3,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[ResearchContext]) -> str:
        """Select system prompt based on language setting."""
        return RESEARCHER_PROMPTS.get(ctx.deps.language, RESEARCHER_PROMPTS["en"])

    return agent


def _build_user_message(ctx: StoryContext) -> str:
    """Build the user message with all pre-fetched context.

    Args:
        ctx: StoryContext with all gathered information

    Returns:
        Formatted message string for the researcher
    """
    story = ctx.story
    publishers = ", ".join(story.publishers) if story.publishers else "Unknown"

    # Build classification context
    classification_lines = []
    if ctx.classification:
        classification_lines.append(f"Category: {ctx.classification.category.value}")
        if ctx.classification.reasoning:
            classification_lines.append(f"Classification: {ctx.classification.reasoning}")
    classification_context = "\n".join(classification_lines)

    # Build article content section
    if ctx.article_content:
        article_section = f"""## Full Article Content
{ctx.article_content[:8000]}
{"... [truncated]" if len(ctx.article_content) > 8000 else ""}"""
    else:
        article_section = """## Full Article Content
(Article content could not be fetched. Analyze based on the description and use web search for verification.)"""

    # Build related stories section
    if ctx.related_stories and ctx.related_stories != "No related stories found in database.":
        related_section = f"""## Related Past Coverage (from our database)
{ctx.related_stories}"""
    else:
        related_section = """## Related Past Coverage
No related stories found in our database. This may be a new topic."""

    message = f"""Analyze this news story:

## Story Information
Title: {story.title}
Publisher: {publishers}
Published: {story.pub_date.strftime("%Y-%m-%d %H:%M")}
{classification_context}

## RSS Description
{story.description[:2000] if story.description else "No description available"}

{article_section}

{related_section}

## Instructions
1. Read and understand the provided content
2. Use web search to verify key claims and gather current context
3. Consider the historical context from related stories
4. Provide a comprehensive analysis following the output structure"""

    return message


class ResearcherAgent:
    """Performs deep analysis on important news stories using Gemini with grounding.

    This agent is the second stage of the analysis pipeline. It receives
    stories that were classified as important along with pre-fetched context
    (article content, related stories) and produces comprehensive research reports.

    Architecture:
        - Single Gemini API call per story
        - WebSearchTool for real-time Google Search grounding
        - Pre-fetched context passed in prompt (no custom tools)

    Example:
        >>> researcher = ResearcherAgent(config)
        >>> context = StoryContext(
        ...     story=story,
        ...     classification=classification,
        ...     article_content=fetched_article,
        ...     related_stories=rag_results,
        ... )
        >>> report = await researcher.analyze(context)
        >>> print(report.summary)
    """

    def __init__(self, config: Config):
        """Initialize the researcher agent.

        Args:
            config: Application configuration with model and language settings
        """
        self.config = config
        self._agent = _create_agent(config.researcher_model)
        self._context = ResearchContext(language=config.language)

    async def analyze(self, story_context: StoryContext) -> ResearchReport:
        """Analyze a story with pre-fetched context.

        Args:
            story_context: StoryContext with story and all pre-fetched information

        Returns:
            Research report with summary and analysis
        """
        story = story_context.story
        message = _build_user_message(story_context)

        try:
            logger.info("Analysis started | title=%s", story.title[:60])
            result = await self._agent.run(
                message,
                deps=self._context,
                usage_limits=UsageLimits(request_limit=5),  # Reduced from 15 (no tool round-trips)
            )
            # Log token usage
            usage = result.usage()
            logger.info(
                "Analysis complete | title=%s chars=%d requests=%d input_tokens=%d output_tokens=%d",
                story.title[:50],
                len(result.output.summary),
                usage.requests,
                usage.request_tokens or 0,
                usage.response_tokens or 0,
            )
            return result.output
        except Exception as e:
            logger.error("Analysis failed for '%s...': %s | type=%s", story.title[:50], e, type(e).__name__, exc_info=True)
            return ResearchReport.empty(reason=f"Analysis error ({type(e).__name__}): {e}")

    async def analyze_batch(
        self,
        story_contexts: list[StoryContext],
        max_concurrent: int = 3,
    ) -> list[tuple[Story, ResearchReport]]:
        """Analyze multiple stories concurrently.

        Args:
            story_contexts: List of StoryContext objects with pre-fetched data
            max_concurrent: Max concurrent API calls

        Returns:
            List of (story, report) tuples
        """
        total = len(story_contexts)
        completed = 0
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info("Batch analysis started | total=%d max_concurrent=%d", total, max_concurrent)

        async def analyze_one(
            index: int,
            story_ctx: StoryContext,
        ) -> tuple[Story, ResearchReport]:
            nonlocal completed
            async with semaphore:
                logger.info("Processing story %d/%d | title=%s", index + 1, total, story_ctx.story.title[:50])
                result = await self.analyze(story_ctx)
                completed += 1
                logger.info("Progress: %d/%d complete (%.0f%%)", completed, total, completed / total * 100)
                return story_ctx.story, result

        tasks = [analyze_one(i, ctx) for i, ctx in enumerate(story_contexts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch analysis error for story %d: %s", i, result, exc_info=result)
                output.append((story_contexts[i].story, ResearchReport.empty()))
            else:
                output.append(result)

        logger.info("Batch analysis complete | total=%d errors=%d", total, sum(1 for r in results if isinstance(r, Exception)))
        return output
