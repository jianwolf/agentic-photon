"""Digest report models for per-run summaries."""

from pydantic import BaseModel, Field


class DigestStory(BaseModel):
    """Single story summary within a run digest."""

    title: str = Field(description="Story title")
    takeaway: str = Field(description="1-2 sentence key takeaway")
    source: str = Field(default="", description="Primary source or outlet (optional)")
    report_file: str = Field(default="", description="Report filename for reference")


class DigestReport(BaseModel):
    """Structured digest output for a pipeline run."""

    overview: str = Field(description="1-2 paragraph overview of the run")
    story_summaries: list[DigestStory] = Field(
        default_factory=list,
        description="One entry per report with concise takeaway",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Cross-cutting themes or trends (3-5 items)",
    )
    watchlist: list[str] = Field(
        default_factory=list,
        description="Follow-up signals/questions to monitor (3-5 items)",
    )
