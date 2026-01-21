"""Classification result models for news importance.

This module defines the importance categories and classification results
used by the classifier agent to determine which stories warrant deep analysis.

Category Design:
    Categories are divided into two groups:

    IMPORTANT (triggers deep analysis):
        Stories about politics, economics, technology, AI/ML, research,
        cybersecurity, international affairs, and policy changes.

    NOT IMPORTANT (skipped):
        Entertainment, sports, lifestyle, local news, weather, and
        general/uncategorized content.

    The classifier agent outputs both a boolean (is_important) and a
    category. Stories marked important are sent to the researcher agent.
"""

from enum import Enum
from pydantic import BaseModel, Field


class ImportanceCategory(str, Enum):
    """Categories for news classification.

    Each category represents a distinct topic area. Categories are divided
    into "important" (warrant deep analysis) and "not important" (can be skipped).

    Important Categories:
        POLITICS: Government, elections, legislation, political figures
        ECONOMICS: Markets, economic indicators, monetary policy
        BUSINESS: Companies, mergers, earnings, industry trends
        INTERNATIONAL: Global affairs, diplomacy, foreign relations
        POLICY: Regulatory changes, new laws, government programs
        TECHNOLOGY: Tech industry, products, platforms, infrastructure
        AI_ML: Artificial intelligence, machine learning, LLMs, models
        RESEARCH: Scientific papers, discoveries, academic work
        SECURITY: Cybersecurity, data breaches, vulnerabilities

    Not Important Categories:
        ENTERTAINMENT: Movies, TV, celebrities, pop culture
        SPORTS: Games, athletes, teams, competitions
        LIFESTYLE: Health, food, travel, personal finance tips
        LOCAL: Regional news, community events
        WEATHER: Forecasts, severe weather
        OTHER: Uncategorized or unclear content
    """

    # === Important Categories (trigger deep analysis) ===
    POLITICS = "politics"           # Government, elections, legislation
    ECONOMICS = "economics"         # Markets, indicators, monetary policy
    BUSINESS = "business"           # Companies, M&A, industry trends
    INTERNATIONAL = "international" # Global affairs, diplomacy
    POLICY = "policy"               # Regulatory changes, new laws
    TECHNOLOGY = "technology"       # Tech industry, products, platforms
    AI_ML = "ai_ml"                 # AI, ML, LLMs, neural networks
    RESEARCH = "research"           # Scientific papers, discoveries
    SECURITY = "security"           # Cybersecurity, breaches, vulnerabilities

    # === Not Important Categories (skip analysis) ===
    ENTERTAINMENT = "entertainment" # Movies, TV, celebrities
    SPORTS = "sports"               # Games, athletes, teams
    LIFESTYLE = "lifestyle"         # Health, food, travel
    LOCAL = "local"                 # Regional/community news
    WEATHER = "weather"             # Forecasts, severe weather
    OTHER = "other"                 # Uncategorized content


# Categories that warrant deep analysis by the researcher agent
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
    """Check if a category is considered important.

    Args:
        category: The category to check

    Returns:
        True if the category triggers deep analysis
    """
    return category in IMPORTANT_CATEGORIES


class ClassificationResult(BaseModel):
    """Result of classifying a news story's importance.

    This model is the output of the classifier agent. It determines whether
    a story should be processed by the researcher agent for deep analysis.

    Attributes:
        is_important: Whether the story warrants deep analysis
        confidence: Model's confidence in the classification (0.0 to 1.0)
        category: Primary topic category from ImportanceCategory enum
        reasoning: Brief explanation of the classification decision

    Example:
        >>> result = ClassificationResult.analyze(
        ...     category=ImportanceCategory.AI_ML,
        ...     confidence=0.95,
        ...     reasoning="Major AI model release announcement"
        ... )
        >>> result.is_important
        True
    """

    is_important: bool = Field(description="Whether story warrants deep analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence (0-1)")
    category: ImportanceCategory = Field(description="Primary topic category")
    reasoning: str = Field(default="", description="Brief explanation of decision")

    @classmethod
    def skip(
        cls,
        category: ImportanceCategory = ImportanceCategory.OTHER,
        reasoning: str = "Does not meet importance criteria",
    ) -> "ClassificationResult":
        """Create a result indicating the story should be skipped.

        Use this factory method for stories that don't warrant deep analysis.

        Args:
            category: The topic category (defaults to OTHER)
            reasoning: Explanation for skipping

        Returns:
            ClassificationResult with is_important=False
        """
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
        """Create a result indicating the story should be analyzed.

        Use this factory method for stories that warrant deep analysis
        by the researcher agent.

        Args:
            category: The topic category (required)
            confidence: Classification confidence (0-1)
            reasoning: Explanation for the decision

        Returns:
            ClassificationResult with is_important=True
        """
        return cls(
            is_important=True,
            confidence=confidence,
            category=category,
            reasoning=reasoning,
        )
