"""Classifier agent for determining news story importance."""

import asyncio
import logging
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from config import Config
from models.story import Story
from models.classification import ClassificationResult, ImportanceCategory

logger = logging.getLogger(__name__)

# System prompts by language
CLASSIFIER_PROMPTS = {
    "zh": """你是一个新闻分类专家。判断新闻是否重要。

**重要**（is_important=true）:
- 政治、经济、商业、国际关系
- 政策、法规变化
- 科技、AI/ML、研究突破
- 网络安全、数据隐私

**不重要**（is_important=false）:
- 娱乐、体育、生活方式
- 地方新闻、天气
- 软文、广告

分析标题和来源，输出JSON分类结果。""",

    "en": """You are a news classification expert. Determine if news is important.

**Important** (is_important=true):
- Politics, economics, business, international
- Policy and regulatory changes
- Technology, AI/ML, research breakthroughs
- Cybersecurity, data privacy

**Not important** (is_important=false):
- Entertainment, sports, lifestyle
- Local news, weather
- Sponsored content, ads

Analyze title and source, output JSON classification.""",
}


@dataclass
class ClassifierContext:
    """Context passed to the classifier agent."""
    language: str = "en"


def _create_agent(model: str) -> Agent[ClassifierContext, ClassificationResult]:
    """Create the underlying PydanticAI agent."""
    agent = Agent(
        model,
        result_type=ClassificationResult,
        system_prompt=CLASSIFIER_PROMPTS["en"],
        retries=3,
    )

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext[ClassifierContext]) -> str:
        return CLASSIFIER_PROMPTS.get(ctx.deps.language, CLASSIFIER_PROMPTS["en"])

    return agent


class ClassifierAgent:
    """Classifies news stories by importance.

    Uses a fast model to quickly determine if stories warrant
    deeper analysis by the researcher agent.
    """

    def __init__(self, config: Config):
        self.config = config
        self._agent = _create_agent(config.classifier_model)
        self._context = ClassifierContext(language=config.language)

    async def classify(self, story: Story) -> ClassificationResult:
        """Classify a single story.

        Args:
            story: Story to classify

        Returns:
            Classification result
        """
        publishers = ", ".join(story.publishers) if story.publishers else "Unknown"
        message = f"""Title: {story.title}
Publisher: {publishers}
Published: {story.pub_date.strftime("%Y-%m-%d %H:%M")}"""

        try:
            result = await self._agent.run(message, deps=self._context)
            logger.debug(f"Classified: {story.title[:50]}... -> {result.data.is_important}")
            return result.data
        except Exception as e:
            logger.error(f"Classification failed for '{story.title[:50]}...': {e}")
            # Default to important on error to avoid missing stories
            return ClassificationResult.analyze(
                category=ImportanceCategory.OTHER,
                confidence=0.5,
                reasoning=f"Classification error: {e}",
            )

    async def classify_batch(
        self,
        stories: list[Story],
        max_concurrent: int = 5,
    ) -> list[tuple[Story, ClassificationResult]]:
        """Classify multiple stories concurrently.

        Args:
            stories: Stories to classify
            max_concurrent: Max concurrent API calls

        Returns:
            List of (story, classification) tuples
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def classify_one(story: Story) -> tuple[Story, ClassificationResult]:
            async with semaphore:
                return story, await self.classify(story)

        tasks = [classify_one(s) for s in stories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch error: {result}")
                output.append((
                    stories[i],
                    ClassificationResult.analyze(
                        category=ImportanceCategory.OTHER,
                        confidence=0.5,
                        reasoning=f"Batch error: {result}",
                    ),
                ))
            else:
                output.append(result)

        return output
