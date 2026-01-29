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
import random
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError

from config import Config
from models.story import Story
from models.classification import ClassificationResult
from models.research import ResearchReport

logger = logging.getLogger(__name__)

# Maximum characters of article content to include in prompt
MAX_ARTICLE_CONTENT_CHARS = 8000
MAX_DESCRIPTION_CHARS = 2000


# === System Prompts ===
# Bilingual prompts instructing the model on analysis methodology.
# Context is now provided in the user message, not via tools.

TECH_RESEARCHER_PROMPTS = {
    "zh": """【语言要求】所有输出内容必须使用中文撰写。技术术语（如AI、LLM、API等）可保留英文。

你是一名资深新闻研究员和事实核查专家，目标是写出**连贯、有洞见、可验证**的新闻分析。

## 你将收到的信息

1. **新闻标题和元数据** - 来自RSS订阅源
2. **文章全文** - 已获取的原始内容（如果可用）
3. **相关历史报道** - 来自我们数据库的过往分析

## 关键原则（必须遵守）

1. **必须使用网络搜索**进行事实核查与补充背景（建议至少2-5条查询）。
2. **只呈现已核实的信息**：summary 与 key_points 只能包含经过搜索验证或权威来源明确支持的内容。
3. **每个关键事实必须附带来源链接**：在 summary / key_points 中用 [1][2] 形式标注；在 thought 末尾提供对应的 Sources 列表（编号 + 来源名 + URL）。
4. **无法核实的内容**只能写在 thought 的“未核实/冲突”部分，summary / key_points 不得包含。
5. **清晰区分**事实、分析、推断，避免把推断写成事实。

## 深度与洞见要求
- 解释因果链条、利益相关方与激励、二阶影响（市场、政策、社会、技术等）。
- 提出1-2个**基于证据**的创造性洞察或可能情景，并说明支撑证据。
- 注重逻辑连贯与段落衔接，避免碎片化。

## 分析流程

### 第一步：理解内容
- 仔细阅读提供的文章全文
- 标记关键声明、数据和引用

### 第二步：网络搜索验证
- 用搜索验证关键声明与数据
- 补充背景与多方观点
- 获取最新进展与权威信息

### 第三步：历史关联
- 结合相关历史报道建立时间线
- 识别是否为持续性事件的最新进展

### 第四步：综合分析
- 仅使用已核实信息输出分析
- 对不确定内容做明确标注（仅放入 thought）

## 输出要求（必须用中文）

### summary（约600-900字，中文）
结构：
1. 导语（1-2句）：核心新闻事实
2. 背景（1-2段）：重要性与背景
3. 详细分析（2-3段）：因果与影响
4. 影响展望（1段）：潜在后续发展
要求：每段至少包含一个来源标注 [n]。

### thought（中文）
必须包含：
- 使用的搜索查询列表
- 关键声明核查结果（可用“声明-证据-结论”简表）
- 来源可信度简评
- 未核实/冲突信息（如有）
- Sources 列表（示例：[1] 来源名 - URL）

### key_points（3-5条，中文）
- 每条独立成立，强调“这意味着什么”
- 每条都要有 [n] 来源标注
- 仅包含已核实信息

### related_topics（中文）
- 值得后续关注的话题或指标
- 可包含情景/信号，但需与已核实事实相关

【再次提醒】你的所有输出（summary、thought、key_points、related_topics）必须使用中文。""",

    "en": """You are a senior news researcher and fact-checker. Your goal is to deliver a **coherent, insightful, and verifiable** analysis.

## Information You Will Receive

1. **News title and metadata** - From RSS feed
2. **Full article content** - Pre-fetched original content (if available)
3. **Related past coverage** - Previous analyses from our database

## Non-Negotiable Principles

1. **You must use web search** to verify facts and add context (aim for 2-5 queries).
2. **Only present verified information**: summary and key_points can include only claims supported by search results or authoritative sources.
3. **Attach source links for every key fact**: use [1][2] markers in summary / key_points, and provide a Sources list in thought (number + outlet name + URL).
4. **Unverified or conflicting claims** may appear only in thought, never in summary / key_points.
5. **Clearly separate** facts, analysis, and inference.

## Depth & Insight Expectations
- Explain causal chains, stakeholder incentives, and second-order effects (market, policy, society, tech).
- Offer 1-2 creative, evidence-based insights or plausible scenarios, and cite supporting evidence.
- Keep narrative flow tight and coherent; avoid fragmented bulleting in the summary.

## Analysis Workflow

### Step 1: Understand Content
- Carefully read the provided article content
- Flag key claims, data, and quotes

### Step 2: Web Search Verification
- Verify key claims and data with search
- Gather background and alternative perspectives
- Capture latest updates from authoritative sources

### Step 3: Historical Context
- Review related stories to build a timeline
- Identify whether this is a new event or the latest update

### Step 4: Synthesize Analysis
- Use only verified information in the report
- Explicitly log uncertainty in thought

## Output Requirements

### summary (~600-900 words)
Structure:
1. Lead (1-2 sentences): Core news fact
2. Background (1-2 paragraphs): Why this matters
3. Detailed Analysis (2-3 paragraphs): Causal and strategic examination
4. Implications (1 paragraph): Potential future developments
Requirement: each paragraph must include at least one source marker [n].

### thought
Must include:
- Search queries used
- Verification log (claim → evidence → conclusion)
- Source credibility assessment
- Unverified/conflicting items (if any)
- Sources list (e.g., [1] Outlet - URL)

### key_points (3-5 items)
- Each stands alone and emphasizes “what this means”
- Each includes [n] source markers
- Verified information only

### related_topics
- Follow-up topics or indicators to monitor
- May include scenarios/signals tied to verified facts""",
}

RESEARCH_RESEARCHER_PROMPTS = {
    "zh": """【语言要求】所有输出内容必须使用中文撰写。技术术语可保留英文。

你是一名资深研究论文分析师与事实核查专家，目标是为**AI/ML 研究论文**写出清晰、结构化、可验证的解读。

## 你将收到的信息

1. **论文标题和元数据** - 来自RSS订阅源
2. **文章全文** - 可能包含论文摘要或正文（若可获取）
3. **相关历史报道** - 来自我们数据库的过往分析

## 关键原则（必须遵守）

1. **必须使用网络搜索**核查关键结论与背景（建议至少2-5条查询）。
2. **仅输出可验证信息**：summary / key_points 只能包含有来源支持的内容。
3. **每个关键事实必须附带来源链接**：在 summary / key_points 中用 [1][2] 形式标注；在 thought 末尾提供 Sources 列表。
4. **不确定或冲突信息**只能写在 thought 中，summary / key_points 禁止包含。
5. **清晰区分**事实、分析、推断。

## 论文解读要点（必须覆盖）
- 研究问题：作者试图解决什么问题？
- 方法/模型：关键技术路线、数据、实验设置
- 结果：主要实验结论、性能或发现
- 局限性：假设、数据偏差、未覆盖的风险
- 意义与影响：对领域或产业可能带来的变化

## 输出要求（必须用中文）

### summary（约600-900字，中文）
结构：
1. 导语：论文核心结论与贡献
2. 方法与实验：关键方法与评估方式
3. 结果与意义：主要结论及影响
4. 局限与展望：限制与后续方向
要求：每段至少包含一个来源标注 [n]。

### thought（中文）
必须包含：
- 使用的搜索查询列表
- 关键声明核查结果（可用“声明-证据-结论”简表）
- 来源可信度简评
- 未核实/冲突信息（如有）
- Sources 列表（示例：[1] 来源名 - URL）

### key_points（3-5条，中文）
- 每条独立成立，强调“这意味着什么”
- 每条都要有 [n] 来源标注
- 仅包含已核实信息

### related_topics（中文）
- 值得后续关注的话题或指标""",

    "en": """You are a senior research-paper analyst and fact-checker. Your goal is to produce a **clear, structured, and verifiable** analysis of AI/ML research papers.

## Information You Will Receive

1. **Paper title and metadata** - From RSS feeds
2. **Full article content** - May include abstract or paper text (if available)
3. **Related past coverage** - Previous analyses from our database

## Non-Negotiable Principles

1. **You must use web search** to verify key claims and add context (aim for 2-5 queries).
2. **Only present verified information**: summary and key_points must be supported by sources.
3. **Attach source links for every key fact**: use [1][2] markers in summary / key_points, and provide a Sources list in thought.
4. **Unverified or conflicting claims** may appear only in thought, never in summary / key_points.
5. **Clearly separate** facts, analysis, and inference.

## Paper Analysis Checklist (must cover)
- Problem: What does the paper aim to solve?
- Method: Core approach, data, and experimental setup
- Results: Key findings or performance improvements
- Limitations: Assumptions, biases, missing cases
- Impact: Implications for the field or industry

## Output Requirements

### summary (~600-900 words)
Structure:
1. Lead: Core contribution and headline result
2. Method & Evaluation: Key method and how it was tested
3. Results & Impact: Main findings and why they matter
4. Limitations & Outlook: Constraints and next steps
Requirement: each paragraph must include at least one source marker [n].

### thought
Must include:
- Search queries used
- Verification log (claim → evidence → conclusion)
- Source credibility assessment
- Unverified/conflicting items (if any)
- Sources list (e.g., [1] Outlet - URL)

### key_points (3-5 items)
- Each stands alone and emphasizes “what this means”
- Each includes [n] source markers
- Verified information only

### related_topics
- Follow-up topics or indicators to monitor""",
}


@dataclass
class ResearchContext:
    """Runtime context passed to the researcher agent.

    Attributes:
        language: Output language ('zh' or 'en')
        track: Analysis track ('tech' or 'research')
    """
    language: str = "en"
    track: str = "tech"


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
        system_prompt=TECH_RESEARCHER_PROMPTS["en"],  # Default fallback
        builtin_tools=[WebSearchTool()],  # Google Search grounding
        retries=3,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[ResearchContext]) -> str:
        """Select system prompt based on language setting."""
        track = ctx.deps.track
        if track == "research":
            prompts = RESEARCH_RESEARCHER_PROMPTS
        else:
            prompts = TECH_RESEARCHER_PROMPTS
        return prompts.get(ctx.deps.language, prompts["en"])

    return agent


def _build_user_message(ctx: StoryContext, language: str = "en", track: str = "tech") -> str:
    """Build the user message with all pre-fetched context.

    Args:
        ctx: StoryContext with all gathered information
        language: Output language ('zh' or 'en')

    Returns:
        Formatted message string for the researcher
    """
    story = ctx.story
    publishers = ", ".join(story.publishers) if story.publishers else "Unknown"

    # Language-specific templates
    if language == "zh":
        templates = {
            "analyze": "请分析以下新闻：",
            "story_info": "## 新闻信息",
            "title": "标题",
            "publisher": "来源",
            "published": "发布时间",
            "category": "分类",
            "track": "路线",
            "classification": "分类说明",
            "description": "## RSS摘要",
            "no_description": "无摘要",
            "article_header": "## 文章全文",
            "article_unavailable": "（文章内容获取失败。请根据摘要分析，并使用网络搜索验证。）",
            "truncated": "... [已截断]",
            "related_header": "## 相关历史报道（来自数据库）",
            "no_related": "数据库中未找到相关报道。这可能是一个新话题。",
            "instructions": "## 分析要求",
            "instruction_list": """1. 阅读并理解提供的内容
2. 使用网络搜索验证关键声明并获取最新背景
3. 结合相关历史报道进行分析
4. 按照输出格式提供完整分析

【重要】请用中文撰写所有输出内容。""",
        }
    else:
        templates = {
            "analyze": "Analyze this news story:",
            "story_info": "## Story Information",
            "title": "Title",
            "publisher": "Publisher",
            "published": "Published",
            "category": "Category",
            "track": "Track",
            "classification": "Classification",
            "description": "## RSS Description",
            "no_description": "No description available",
            "article_header": "## Full Article Content",
            "article_unavailable": "(Article content could not be fetched. Analyze based on the description and use web search for verification.)",
            "truncated": "... [truncated]",
            "related_header": "## Related Past Coverage (from our database)",
            "no_related": "No related stories found in our database. This may be a new topic.",
            "instructions": "## Instructions",
            "instruction_list": """1. Read and understand the provided content
2. Use web search to verify key claims and gather current context
3. Consider the historical context from related stories
4. Provide a comprehensive analysis following the output structure""",
        }

    # Build classification context
    classification_lines = []
    if ctx.classification:
        classification_lines.append(f"{templates['category']}: {ctx.classification.category.value}")
        if ctx.classification.reasoning:
            classification_lines.append(f"{templates['classification']}: {ctx.classification.reasoning}")
    classification_context = "\n".join(classification_lines)
    if language == "zh":
        track_label = "研究论文" if track == "research" else "科技新闻"
    else:
        track_label = "Research Paper" if track == "research" else "Tech News"

    # Build article content section
    if ctx.article_content:
        content = ctx.article_content[:MAX_ARTICLE_CONTENT_CHARS]
        truncated = templates["truncated"] if len(ctx.article_content) > MAX_ARTICLE_CONTENT_CHARS else ""
        article_section = f"{templates['article_header']}\n{content}{truncated}"
    else:
        article_section = f"{templates['article_header']}\n{templates['article_unavailable']}"

    # Build related stories section
    if ctx.related_stories and ctx.related_stories != "No related stories found in database.":
        related_section = f"{templates['related_header']}\n{ctx.related_stories}"
    else:
        related_section = f"{templates['related_header']}\n{templates['no_related']}"

    message = f"""{templates['analyze']}

{templates['story_info']}
{templates['title']}: {story.title}
{templates['publisher']}: {publishers}
{templates['published']}: {story.pub_date.strftime("%Y-%m-%d %H:%M")}
{templates['track']}: {track_label}
{classification_context}

{templates['description']}
{story.description[:MAX_DESCRIPTION_CHARS] if story.description else templates['no_description']}

{article_section}

{related_section}

{templates['instructions']}
{templates['instruction_list']}"""

    return message


def _is_retryable_gemini_error(error: Exception) -> bool:
    if not isinstance(error, ModelHTTPError):
        return False
    model_name = getattr(error, "model_name", "") or ""
    if "gemini" not in model_name:
        return False
    status_code = getattr(error, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {}) if isinstance(body.get("error", {}), dict) else {}
        message = str(err.get("message", "")).lower()
        status = str(err.get("status", "")).upper()
        if "overloaded" in message or status in {"UNAVAILABLE", "RESOURCE_EXHAUSTED"}:
            return True
    return False


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

    def __init__(self, config: Config, track: str = "tech"):
        """Initialize the researcher agent.

        Args:
            config: Application configuration with model and language settings
            track: Analysis track ('tech' or 'research')
        """
        self.config = config
        self._agent = _create_agent(config.researcher_model)
        self._context = ResearchContext(language=config.language, track=track)

    async def analyze(self, story_context: StoryContext) -> tuple[ResearchReport, int, int]:
        """Analyze a story with pre-fetched context.

        Args:
            story_context: StoryContext with story and all pre-fetched information

        Returns:
            Tuple of (research report, input_tokens, output_tokens)
        """
        story = story_context.story
        message = _build_user_message(
            story_context,
            language=self._context.language,
            track=self._context.track,
        )

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                result = await self._agent.run(
                    message,
                    deps=self._context,
                    usage_limits=UsageLimits(request_limit=5),  # Reduced from 15 (no tool round-trips)
                )
                # Log token usage
                usage = result.usage()
                input_tokens = usage.request_tokens or 0
                output_tokens = usage.response_tokens or 0
                logger.info(
                    "Analysis complete | title=%s chars=%d input_tokens=%d output_tokens=%d",
                    story.title[:50],
                    len(result.output.summary),
                    input_tokens,
                    output_tokens,
                )
                return result.output, input_tokens, output_tokens
            except Exception as e:
                if _is_retryable_gemini_error(e) and attempt < max_retries:
                    wait_seconds = min(30.0, (2 ** attempt) + random.uniform(0.5, 1.5))
                    logger.warning(
                        "Gemini overloaded; retrying | attempt=%d/%d wait=%.1fs title=%s error=%s",
                        attempt + 1,
                        max_retries + 1,
                        wait_seconds,
                        story.title[:50],
                        e,
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                logger.error("Analysis failed for '%s...': %s | type=%s", story.title[:50], e, type(e).__name__, exc_info=True)
                return ResearchReport.empty(reason=f"Analysis error ({type(e).__name__}): {e}"), 0, 0

    async def analyze_batch(
        self,
        story_contexts: list[StoryContext],
        max_concurrent: int = 3,
    ) -> tuple[list[tuple[Story, ResearchReport]], int, int]:
        """Analyze multiple stories concurrently.

        Args:
            story_contexts: List of StoryContext objects with pre-fetched data
            max_concurrent: Max concurrent API calls

        Returns:
            Tuple of (list of (story, report) tuples, total_input_tokens, total_output_tokens)
        """
        total = len(story_contexts)
        completed = 0
        total_input_tokens = 0
        total_output_tokens = 0
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info("Batch analysis started | total=%d max_concurrent=%d", total, max_concurrent)

        async def analyze_one(
            index: int,
            story_ctx: StoryContext,
        ) -> tuple[Story, ResearchReport, int, int]:
            nonlocal completed, total_input_tokens, total_output_tokens
            async with semaphore:
                logger.debug("Processing story %d/%d | title=%s", index + 1, total, story_ctx.story.title[:50])
                report, input_tokens, output_tokens = await self.analyze(story_ctx)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                completed += 1
                logger.info("Progress: %d/%d complete (%.0f%%)", completed, total, completed / total * 100)
                return story_ctx.story, report, input_tokens, output_tokens

        tasks = [analyze_one(i, ctx) for i, ctx in enumerate(story_contexts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        errors = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch analysis error for story %d: %s", i, result, exc_info=result)
                output.append((story_contexts[i].story, ResearchReport.empty()))
                errors += 1
            else:
                story, report, _, _ = result
                output.append((story, report))

        logger.info(
            "Batch analysis complete | total=%d errors=%d input_tokens=%d output_tokens=%d",
            total, errors, total_input_tokens, total_output_tokens
        )
        return output, total_input_tokens, total_output_tokens
