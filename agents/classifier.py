"""Classifier agent for determining news story importance.

This module implements the ClassifierAgent, which uses a fast AI model to
quickly determine whether a news story warrants deep analysis.

Design Philosophy:
    - Speed over depth: Uses a fast model for quick triage
    - Fail-safe: Defaults to "important" on errors (avoid missing stories)
    - Concurrent: Batch classification with semaphore-controlled parallelism

The classifier outputs a ClassificationResult with:
    - is_important: Boolean gate for the researcher agent
    - category: Topic classification for context
    - confidence: Model's certainty (0-1)
    - reasoning: Brief explanation
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from config import Config
from models.story import Story
from models.classification import ClassificationResult, ImportanceCategory

logger = logging.getLogger(__name__)

MAX_CLASSIFIER_INPUT_CHARS = 80_000
MAX_CLASSIFIER_OUTPUT_TOKENS = 160

# === System Prompts ===
# Bilingual prompts for Chinese (zh) and English (en) output.
# The prompt instructs the model to classify news importance based on topic.

CLASSIFIER_PROMPTS = {
    "zh": """你是一个专业的新闻价值评估分析师。你的任务是快速、准确地判断新闻的重要性。

## 判断标准

**标记为重要 (is_important=true) 的新闻：**
- 政治与国际关系：政策变化、外交事件、选举、政府人事变动
- 经济与金融：重大市场波动、央行决策、企业并购、经济指标
- 科技突破：AI/ML研究进展、重大产品发布、安全漏洞、监管变化
- 社会影响：影响广泛人群的事件、公共卫生、基础设施
- **主要AI公司的产品发布**：OpenAI、Google、Anthropic、Meta、NVIDIA等公司的新产品、新功能、新模型发布（如ChatGPT新功能、Gemini更新、Claude新版本、Veo视频模型等）
- **AI进入新领域**：AI应用于医疗、法律、教育等垂直领域的重大进展（如ChatGPT Health、医疗AI等）
- **行业竞争分析**：关于主要科技/AI公司之间竞争格局、市场地位变化的深度分析
- **AI行业预测与趋势**：对AI/ML行业发展方向的预测、年度总结、趋势分析
- **科技/AI主题的定期专栏**：如"The Pulse"、"AI Weekly"等专注于AI/编程/科技的定期专栏或newsletter
- **互联网基础设施与安全**：BGP异常、DNS问题、大规模网络中断、国家级互联网封锁等

**标记为不重要 (is_important=false) 的新闻：**
- 娱乐八卦、体育赛事结果、名人动态
- 地区性小新闻、天气预报、生活方式内容
- 软文、广告、客户案例研究（如"某公司如何使用ChatGPT"）、榜单推荐
- 个人博客文章、引用汇编、个人学习笔记
- 小型开发工具发布（除非是平台级或改变行业的工具）
- 公司内部公告（如招聘、办公室搬迁）

## 边界情况处理
- 科技公司的商业新闻 → 看是否涉及行业格局变化
- 同一事件的后续报道 → 仅新增实质内容才标记重要
- 来源可疑的"重大新闻" → 降低置信度但仍标记重要

## 置信度校准
- 0.9-1.0：明确符合或不符合标准
- 0.7-0.9：基本符合，有轻微模糊性
- 0.5-0.7：边界案例，需谨慎判断
- 如有疑虑，宁可标记为重要

## 输出要求
分析新闻标题和来源后，输出JSON格式的分类结果，包含：
- is_important: 布尔值
- confidence: 0-1之间的数值
- category: 分类类别
- reasoning: 1-2句简短解释""",

    "en": """You are a professional news value assessment analyst. Your task is to quickly and accurately determine news importance.

## Classification Criteria

**Mark as Important (is_important=true):**
- Politics & International: Policy changes, diplomatic events, elections, government appointments
- Economics & Finance: Major market movements, central bank decisions, M&A, economic indicators
- Technology Breakthroughs: AI/ML research advances, major product launches, security vulnerabilities, regulatory changes
- Societal Impact: Events affecting large populations, public health, infrastructure
- **Major AI Company Product Launches**: New products, features, or models from OpenAI, Google, Anthropic, Meta, NVIDIA, etc. (e.g., ChatGPT features, Gemini updates, Claude versions, Veo video models, etc.)
- **AI Entering New Domains**: Major developments in AI applied to healthcare, legal, education, and other verticals (e.g., ChatGPT Health, medical AI, etc.)
- **Industry Competitive Analysis**: Deep analysis of competition between major tech/AI companies, market position shifts
- **AI Industry Forecasts & Trends**: Predictions about AI/ML industry direction, annual reviews, trend analysis
- **Tech/AI-focused Newsletters & Columns**: Regular columns like "The Pulse", "AI Weekly", etc. that focus on AI/programming/tech topics
- **Internet Infrastructure & Security**: BGP anomalies, DNS issues, large-scale outages, nation-state internet shutdowns

**Mark as Not Important (is_important=false):**
- Entertainment gossip, sports scores, celebrity news
- Local small news, weather forecasts, lifestyle content
- Sponsored content, advertisements, customer case studies (e.g., "How X company uses ChatGPT"), listicles
- Personal blog posts, quote compilations, personal learning notes
- Small developer tool releases (unless platform-level or industry-changing)
- Company internal announcements (hiring, office moves, etc.)

## Edge Case Handling
- Tech company business news → Consider if it changes industry landscape
- Follow-up reports on same event → Only mark important if substantial new information
- "Breaking news" from questionable sources → Lower confidence but still mark important

## Confidence Calibration
- 0.9-1.0: Clearly matches or doesn't match criteria
- 0.7-0.9: Generally matches with minor ambiguity
- 0.5-0.7: Edge case requiring careful judgment
- When in doubt, err toward marking as important

## Output Requirements
After analyzing the title and source, output JSON classification with:
- is_important: boolean
- confidence: number between 0-1
- category: classification category
- reasoning: 1-2 sentence brief explanation""",
}


@dataclass
class ClassifierContext:
    """Runtime context passed to the classifier agent.

    This context is provided as `deps` to the PydanticAI agent and
    is used to select the appropriate language-specific system prompt.

    Attributes:
        language: Output language ('zh' or 'en')
    """
    language: str = "en"


def _parse_local_model(model_str: str) -> tuple[str, str] | None:
    """Parse local model string into (model_name, base_url) or None if not local."""
    if model_str.startswith("openai:") and "@" in model_str:
        rest = model_str[7:]  # Remove "openai:" prefix
        model_name, base_url = rest.split("@", 1)
        return model_name, base_url
    return None


def _create_model(model_str: str):
    """Create the appropriate model based on the model string.

    Supports:
    - Local MLX models: 'openai:{model_name}@http://127.0.0.1:8080/v1'
    - Remote models: 'google-gla:gemini-3-flash-preview'

    Args:
        model_str: Model identifier string

    Returns:
        PydanticAI model instance or model string
    """
    parsed = _parse_local_model(model_str)
    if parsed:
        model_name, base_url = parsed
        logger.info("Using local MLX model | model=%s base_url=%s", model_name, base_url)
        # Local models don't need authentication - use placeholder
        client = AsyncOpenAI(base_url=base_url, api_key="local-model")
        # MLX server doesn't support response_format or tool_choice
        profile = OpenAIModelProfile(supports_json_object_output=False)
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(openai_client=client),
            profile=profile,
        )
    else:
        # Remote model (Gemini, OpenAI, etc.)
        return model_str


def _create_agent(model: str) -> Agent[ClassifierContext, ClassificationResult]:
    """Create the underlying PydanticAI agent for classification.

    The agent uses:
    - Structured output: ClassificationResult Pydantic model
    - Dynamic system prompt: Selected based on language context
    - Retry logic: 3 attempts on failure

    Args:
        model: PydanticAI model string (e.g., 'google-gla:gemini-3-flash-preview')
               or local model string (e.g., 'openai:local@http://127.0.0.1:8080/v1')

    Returns:
        Configured PydanticAI Agent
    """
    model_instance = _create_model(model)

    agent = Agent(
        model_instance,
        # Use PromptedOutput for local models that don't support tool_choice
        output_type=PromptedOutput(ClassificationResult),
        system_prompt=CLASSIFIER_PROMPTS["en"],  # Default fallback
        retries=3,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[ClassifierContext]) -> str:
        """Select system prompt based on language setting."""
        return CLASSIFIER_PROMPTS.get(ctx.deps.language, CLASSIFIER_PROMPTS["en"])

    return agent


class ClassifierAgent:
    """Classifies news stories by importance using AI.

    This agent is the first stage of the analysis pipeline. It quickly
    evaluates each story and determines whether it warrants deeper
    analysis by the researcher agent.

    The agent uses a fast model (typically Gemini Flash) optimized for
    speed over depth. Stories classified as important proceed to the
    researcher; others are saved and skipped.

    Error Handling:
        On classification failure, the agent defaults to marking the story
        as important (fail-safe). This ensures we don't miss potentially
        important stories due to transient API errors.

    Example:
        >>> classifier = ClassifierAgent(config)
        >>> result = await classifier.classify(story)
        >>> if result.is_important:
        ...     # Send to researcher agent
    """

    def __init__(self, config: Config):
        """Initialize the classifier agent.

        Args:
            config: Application configuration with model and language settings
        """
        self.config = config
        self._context = ClassifierContext(language=config.language)

        # Check if using local model (needs special handling)
        self._local_model = _parse_local_model(config.classifier_model)
        if self._local_model:
            model_name, base_url = self._local_model
            self._client = AsyncOpenAI(base_url=base_url, api_key="local-model")
            self._agent = None  # Not used for local models
        else:
            self._client = None
            self._agent = _create_agent(config.classifier_model)

    async def _classify_local(self, story: Story) -> ClassificationResult:
        """Classify using local MLX model with direct API call.

        The local Ministral model doesn't support system messages or
        consecutive messages of the same role, so we use a single
        user message with embedded instructions.

        Uses low temperature (0.1) and top_p (0.1) for deterministic,
        consistent classification results. Chain-of-thought prompting
        improves reasoning quality.
        """
        model_name, _ = self._local_model
        publishers = ", ".join(story.publishers) if story.publishers else "Unknown"

        # Get the detailed system prompt based on language setting
        system_prompt = CLASSIFIER_PROMPTS.get(self._context.language, CLASSIFIER_PROMPTS["en"])

        def _build_prompt(strict: bool, title: str, publishers_str: str) -> str:
            if strict:
                strict_rules = """Return ONLY a single JSON object and nothing else.
- Use double quotes for all strings and keys.
- Escape backslashes (\\\\) and quotes (\\") inside strings.
- Do not include markdown fences or analysis text.
- Output the JSON on a single line."""
                analysis_line = "Provide only the JSON object."
            else:
                strict_rules = ""
                analysis_line = """Think step-by-step:

1. First, identify the main topic and source type
2. Check if it matches any "Important" criteria
3. Check if it matches any "Not Important" criteria
4. Make your final decision
"""
            return f"""{system_prompt}

---

Classify the following news story. {analysis_line}

Then output ONLY valid JSON with these fields:
- is_important: boolean
- confidence: number 0-1
- category: one of [politics, economics, business, technology, ai_ml, research, security, entertainment, sports, lifestyle, other]
- reasoning: brief explanation (1-2 sentences)

{strict_rules if strict else ""}

Title: {title}
Publisher: {publishers_str}
Published: {story.pub_date.strftime("%Y-%m-%d %H:%M")}

JSON:"""

        max_retries = 3
        last_error: Exception | None = None

        timeout_seconds = 60
        for attempt in range(max_retries + 1):
            strict = attempt > 0
            title = story.title
            publishers_str = publishers
            prompt = _build_prompt(strict=strict, title=title, publishers_str=publishers_str)
            if len(prompt) > MAX_CLASSIFIER_INPUT_CHARS:
                overflow = len(prompt) - MAX_CLASSIFIER_INPUT_CHARS
                if overflow > 0 and len(title) > 20:
                    trim = min(overflow, len(title) - 20)
                    title = title[:-trim] + "..."
                    prompt = _build_prompt(strict=strict, title=title, publishers_str=publishers_str)
                    overflow = len(prompt) - MAX_CLASSIFIER_INPUT_CHARS
                if overflow > 0 and len(publishers_str) > 20:
                    trim = min(overflow, len(publishers_str) - 20)
                    publishers_str = publishers_str[:-trim] + "..."
                    prompt = _build_prompt(strict=strict, title=title, publishers_str=publishers_str)
                    overflow = len(prompt) - MAX_CLASSIFIER_INPUT_CHARS
                if overflow > 0:
                    logger.warning(
                        "Classifier prompt truncated | title=%s chars=%d limit=%d",
                        story.title[:50],
                        len(prompt),
                        MAX_CLASSIFIER_INPUT_CHARS,
                    )
                    suffix = "\n\nJSON:"
                    prompt = prompt[: MAX_CLASSIFIER_INPUT_CHARS - len(suffix)] + suffix

            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        top_p=0.1,
                        max_tokens=MAX_CLASSIFIER_OUTPUT_TOKENS,
                        stream=False,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Local classification timed out | attempt=%d/%d title=%s timeout=%ds",
                        attempt + 1,
                        max_retries + 1,
                        story.title[:50],
                        timeout_seconds,
                    )
                    continue
                raise

            content = resp.choices[0].message.content.strip()
            json_str = None

            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                for part in parts[1::2]:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        json_str = part
                        break

            if not json_str:
                start = content.rfind("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = content[start:end]

            if not json_str:
                raise ValueError(f"No JSON found in response: {content[:200]}")

            try:
                data = json.loads(json_str)
                return ClassificationResult(
                    is_important=data.get("is_important", True),
                    confidence=data.get("confidence", 0.5),
                    category=ImportanceCategory(data.get("category", "other")),
                    reasoning=data.get("reasoning", ""),
                )
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Local classification JSON decode failed | attempt=%d/%d title=%s error=%s",
                        attempt + 1,
                        max_retries + 1,
                        story.title[:50],
                        e,
                    )
                    continue
                raise

        raise last_error if last_error else ValueError("Local classification failed without JSON output")

    async def classify(self, story: Story) -> ClassificationResult:
        """Classify a single story.

        Args:
            story: Story to classify

        Returns:
            Classification result
        """
        try:
            if self._local_model:
                result = await self._classify_local(story)
                logger.debug("Classified: %s... -> %s", story.title[:50], result.is_important)
                return result

            # Use pydantic-ai agent for remote models
            publishers = ", ".join(story.publishers) if story.publishers else "Unknown"
            message = f"""Title: {story.title}
Publisher: {publishers}
Published: {story.pub_date.strftime("%Y-%m-%d %H:%M")}"""

            result = await self._agent.run(message, deps=self._context)
            usage = result.usage()
            logger.debug(
                "Classified: %s... -> %s | requests=%d tokens=%d/%d",
                story.title[:50],
                result.output.is_important,
                usage.requests,
                usage.request_tokens or 0,
                usage.response_tokens or 0,
            )
            return result.output
        except Exception as e:
            logger.error("Classification failed for '%s...': %s", story.title[:50], e, exc_info=True)
            # Default to important on error to avoid missing stories
            # Use low confidence (0.3) to indicate this is a fallback decision
            return ClassificationResult.analyze(
                category=ImportanceCategory.OTHER,
                confidence=0.3,
                reasoning=f"Classification error ({type(e).__name__}): {e}",
            )

    def _default_max_concurrent(self) -> int:
        """Default concurrency based on model type."""
        return 3 if self._local_model else 20

    async def classify_batch(
        self,
        stories: list[Story],
        max_concurrent: int | None = None,
    ) -> list[tuple[Story, ClassificationResult]]:
        """Classify multiple stories concurrently.

        Args:
            stories: Stories to classify
            max_concurrent: Max concurrent API calls (None uses model defaults)

        Returns:
            List of (story, classification) tuples
        """
        if max_concurrent is None:
            max_concurrent = self._default_max_concurrent()
        elif self._local_model:
            max_concurrent = min(max_concurrent, self._default_max_concurrent())
        total = len(stories)
        completed = 0
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info("Batch classification started | total=%d max_concurrent=%d", total, max_concurrent)

        async def classify_one(story: Story) -> tuple[Story, ClassificationResult]:
            nonlocal completed
            async with semaphore:
                result = await self.classify(story)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logger.info("Classification progress: %d/%d (%.0f%%)", completed, total, completed / total * 100)
                return story, result

        tasks = [classify_one(s) for s in stories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch classification error for story %d: %s", i, result, exc_info=result)
                output.append((
                    stories[i],
                    ClassificationResult.analyze(
                        category=ImportanceCategory.OTHER,
                        confidence=0.3,
                        reasoning=f"Batch error ({type(result).__name__}): {result}",
                    ),
                ))
            else:
                output.append(result)

        logger.info("Batch classification complete | total=%d errors=%d", total, sum(1 for r in results if isinstance(r, Exception)))
        return output
