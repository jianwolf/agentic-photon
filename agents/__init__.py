"""PydanticAI agents for the Photon news analysis pipeline.

This package contains the two AI agents that power story analysis:

ClassifierAgent:
    Fast importance classification using Gemini Flash.
    Determines which stories warrant deep analysis.

ResearcherAgent:
    Deep analysis with tools for web search, article fetching,
    and history lookup. Produces comprehensive reports.

Example:
    >>> from agents import ClassifierAgent, ResearcherAgent
    >>> classifier = ClassifierAgent(config)
    >>> researcher = ResearcherAgent(config)
"""

from agents.classifier import ClassifierAgent
from agents.researcher import ResearcherAgent

__all__ = [
    "ClassifierAgent",
    "ResearcherAgent",
]
