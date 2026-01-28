"""Pydantic models for the Photon news analysis pipeline.

This package contains all data models used throughout the pipeline:

Story:
    RSS feed item with title, description, pub_date, source_url.
    Includes hash property for deduplication.

ClassificationResult:
    Output of the classifier agent (is_important, category, confidence).

ImportanceCategory:
    Enum of topic categories (POLITICS, AI_ML, TECHNOLOGY, etc.).

ResearchReport:
    Full output of the researcher agent (summary, key_points, etc.).

Analysis:
    Database-compatible subset of ResearchReport for storage.

Example:
    >>> from models import Story, ClassificationResult, ImportanceCategory
    >>> story = Story(title="...", description="...", pub_date=..., source_url="...")
    >>> result = ClassificationResult.analyze(category=ImportanceCategory.AI_ML)
"""

from models.story import Story
from models.classification import ClassificationResult, ImportanceCategory
from models.research import ResearchReport, Analysis
from models.summary import DigestReport, DigestStory

__all__ = [
    "Story",
    "ClassificationResult",
    "ImportanceCategory",
    "ResearchReport",
    "Analysis",
    "DigestReport",
    "DigestStory",
]
