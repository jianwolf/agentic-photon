"""Web search tool for fact-checking and research.

This module provides the web search tool used by the researcher agent.
Currently a placeholder - returns empty results.

INTEGRATION GUIDE:
    To enable real web search, implement one of these backends:

    1. Google Custom Search API:
       - Create project at console.developers.google.com
       - Enable Custom Search API
       - Create a Programmable Search Engine
       - Set GOOGLE_CSE_ID and GOOGLE_API_KEY

    2. Bing Web Search API:
       - Create resource in Azure portal
       - Set BING_SEARCH_KEY

    3. SerpAPI:
       - Sign up at serpapi.com
       - Set SERPAPI_KEY

    4. Gemini with Google Search grounding:
       - Enable search grounding in model configuration
       - No separate API key needed

    The researcher agent will fall back to its training data when
    this tool returns empty results.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResults:
    """Results from a web search query.

    Attributes:
        query: The original search query
        results: List of result dicts with 'title', 'url', 'snippet' keys
    """

    query: str
    results: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Format results as a readable summary for the agent.

        Returns:
            Human-readable string with numbered results
        """
        if not self.results:
            return f"No results found for: {self.query}"

        lines = [f"Search results for: {self.query}\n"]
        for i, r in enumerate(self.results[:5], 1):
            lines.append(f"{i}. {r.get('title', 'Untitled')}")
            if r.get("url"):
                lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                snippet = r["snippet"][:200]
                lines.append(f"   {snippet}...")
            lines.append("")
        return "\n".join(lines)


async def web_search(query: str, num_results: int = 5) -> SearchResults:
    """Search the web for information.

    PLACEHOLDER: Currently returns empty results. The researcher agent
    will rely on its training knowledge when this returns nothing.

    To implement real search, add API calls here and return populated
    SearchResults with title, url, and snippet for each result.

    Args:
        query: Search query string
        num_results: Maximum number of results to return

    Returns:
        SearchResults with empty results list (placeholder)
    """
    logger.debug(f"Web search (placeholder): {query}")

    # TODO: Implement real search backend
    # Example structure for results:
    # results = [
    #     {"title": "Page Title", "url": "https://...", "snippet": "Preview..."},
    # ]
    return SearchResults(query=query, results=[])
