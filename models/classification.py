"""Classification result models for news importance."""

from enum import Enum
from pydantic import BaseModel, Field


class ImportanceCategory(str, Enum):
    """Categories for news classification.

    Categories are divided into "important" (worth analyzing) and
    "not important" (can be skipped).
    """

    # Important categories (will trigger deep analysis)
    POLITICS = "politics"
    ECONOMICS = "economics"
    BUSINESS = "business"
    INTERNATIONAL = "international"
    POLICY = "policy"
    TECHNOLOGY = "technology"
    AI_ML = "ai_ml"
    RESEARCH = "research"
    SECURITY = "security"

    # Not important categories (will be skipped)
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    LIFESTYLE = "lifestyle"
    LOCAL = "local"
    WEATHER = "weather"
    OTHER = "other"


# Categories that warrant deep analysis
IMPORTANT_CATEGORIES: frozenset[ImportanceCategory] = frozenset({
    ImportanceCategory.POLITICS,
    ImportanceCategory.ECONOMICS,
    ImportanceCategory.BUSINESS,
    ImportanceCategory.INTERNATIONAL,
    ImportanceCategory.POLICY,
    ImportanceCategory.TECHNOLOGY,
    ImportanceCategory.AI_ML,
    ImportanceCategory.RESEARCH,
    ImportanceCategory.SECURITY,
})


def is_important_category(category: ImportanceCategory) -> bool:
    """Check if a category is considered important."""
    return category in IMPORTANT_CATEGORIES


class ClassificationResult(BaseModel):
    """Result of classifying a news story's importance.

    Attributes:
        is_important: Whether the story warrants deep analysis
        confidence: Model's confidence in the classification (0-1)
        category: Primary topic category
        reasoning: Brief explanation of the decision
    """

    is_important: bool = Field(description="Whether story warrants analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence (0-1)")
    category: ImportanceCategory = Field(description="Topic category")
    reasoning: str = Field(default="", description="Explanation")

    @classmethod
    def skip(
        cls,
        category: ImportanceCategory = ImportanceCategory.OTHER,
        reasoning: str = "Does not meet importance criteria",
    ) -> "ClassificationResult":
        """Create a result for a story to skip."""
        return cls(
            is_important=False,
            confidence=0.9,
            category=category,
            reasoning=reasoning,
        )

    @classmethod
    def analyze(
        cls,
        category: ImportanceCategory,
        confidence: float = 0.9,
        reasoning: str = "",
    ) -> "ClassificationResult":
        """Create a result for a story to analyze."""
        return cls(
            is_important=True,
            confidence=confidence,
            category=category,
            reasoning=reasoning,
        )
