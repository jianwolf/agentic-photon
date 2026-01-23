"""Research report models for analyzed stories.

This module defines the output models for the researcher agent. After a story
is classified as important, the researcher produces a ResearchReport with
detailed analysis.

Model Hierarchy:
    ResearchReport: Full output from researcher agent (summary, key points, etc.)
    Analysis: Simplified format stored in SQLite (is_important, summary, thought)

The Analysis model is a subset of ResearchReport, used for database storage
and notifications.
"""

from pydantic import BaseModel, Field


class ResearchReport(BaseModel):
    """Research report produced by the researcher agent.

    This is the full output from the researcher agent after analyzing an
    important news story. It includes comprehensive analysis, verification
    notes, and suggested follow-up topics.

    The report is generated using the agent's tools (web search, article
    fetching, database queries) to gather context and verify claims.

    Attributes:
        summary: Detailed analysis with background and context (~600-1000 words)
        thought: Internal notes about sources and fact-checking process
        key_points: 3-5 bullet-point takeaways for quick scanning
        related_topics: Suggested topics for continued monitoring

    Example:
        >>> report = ResearchReport(
        ...     summary="OpenAI released GPT-5, featuring...",
        ...     thought="Verified via official blog and news sources",
        ...     key_points=["New architecture", "Improved reasoning"],
        ...     related_topics=["AGI safety", "Model benchmarks"]
        ... )
    """

    summary: str = Field(
        description="Detailed analysis with background and context (~600-1000 words)"
    )
    thought: str = Field(
        default="",
        description="Source analysis, fact-checking notes, and reasoning process"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="3-5 key takeaways as bullet points"
    )
    related_topics: list[str] = Field(
        default_factory=list,
        description="Related topics to monitor for follow-up"
    )

    @classmethod
    def empty(cls, reason: str = "") -> "ResearchReport":
        """Create an empty report for error cases.

        Used when analysis fails and we need a placeholder report.

        Args:
            reason: Optional error description to store in thought field

        Returns:
            ResearchReport with empty summary and optional error in thought
        """
        return cls(summary="", thought=reason)

    def __str__(self) -> str:
        """Human-readable representation for logging."""
        summary_preview = self.summary[:50] + "..." if len(self.summary) > 50 else self.summary
        return f"ResearchReport('{summary_preview}')"


class Analysis(BaseModel):
    """Database-compatible analysis result.

    This is the simplified format stored in SQLite. It contains only the
    essential fields needed for storage and notifications.

    The Analysis model is created either:
    - From a ResearchReport (for analyzed important stories)
    - As empty (for skipped non-important stories)

    Attributes:
        is_important: Whether the story received deep analysis
        summary: Story summary (from ResearchReport, or empty if skipped)
        thought: Verification notes (from ResearchReport, or empty if skipped)
    """

    is_important: bool = Field(description="Whether story received deep analysis")
    summary: str = Field(default="", description="Detailed summary from researcher")
    thought: str = Field(default="", description="Analysis notes and verification")

    @classmethod
    def empty(cls) -> "Analysis":
        """Create an empty analysis for non-important stories.

        Used when a story is classified as not important and skipped.

        Returns:
            Analysis with is_important=False and empty fields
        """
        return cls(is_important=False, summary="", thought="")

    @classmethod
    def from_report(cls, report: ResearchReport) -> "Analysis":
        """Create an Analysis from a ResearchReport.

        Extracts the summary and thought fields from the full report
        for database storage.

        Args:
            report: The full ResearchReport from the researcher agent

        Returns:
            Analysis with is_important=True and fields from report
        """
        return cls(
            is_important=True,
            summary=report.summary,
            thought=report.thought,
        )
