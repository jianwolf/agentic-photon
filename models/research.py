"""Research report models for analyzed stories."""

from pydantic import BaseModel, Field


class ResearchReport(BaseModel):
    """Research report produced by the researcher agent.

    Contains the detailed analysis of an important news story,
    including summary, verification notes, and key takeaways.
    """

    summary: str = Field(
        description="Detailed summary with background and context"
    )
    thought: str = Field(
        default="",
        description="Source analysis and fact-checking notes"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Key takeaways (3-5 bullet points)"
    )
    related_topics: list[str] = Field(
        default_factory=list,
        description="Related topics for follow-up research"
    )

    @classmethod
    def empty(cls) -> "ResearchReport":
        """Create an empty report."""
        return cls(summary="", thought="")


class Analysis(BaseModel):
    """Database-compatible analysis result.

    This is the format stored in SQLite, containing the essential
    fields from classification and research.
    """

    is_important: bool = Field(description="Whether story was analyzed")
    summary: str = Field(default="", description="Story summary")
    thought: str = Field(default="", description="Verification notes")

    @classmethod
    def empty(cls) -> "Analysis":
        """Create an empty analysis for skipped stories."""
        return cls(is_important=False, summary="", thought="")

    @classmethod
    def from_report(cls, report: ResearchReport) -> "Analysis":
        """Create from a ResearchReport."""
        return cls(
            is_important=True,
            summary=report.summary,
            thought=report.thought,
        )
