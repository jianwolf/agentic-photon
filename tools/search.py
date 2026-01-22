"""Web search tool for fact-checking and research.

This module provides the web search tool used by the researcher agent.

CURRENT STATUS: Search is disabled (returns empty results).
The researcher agent will rely on article fetching and its training data.

To enable real search, implement one of these backends:
    1. Google Custom Search API (set GOOGLE_CSE_ID and GOOGLE_API_KEY)
    2. SerpAPI (set SERPAPI_KEY)
    3. Bing Web Search API (set BING_SEARCH_KEY)

NOTE: Do NOT use Gemini with search grounding here - that would create
double LLM calls (researcher agent + search tool), wasting tokens.
Use a direct search API instead.
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Check for search API keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

SEARCH_ENABLED = bool(GOOGLE_API_KEY and GOOGLE_CSE_ID) or bool(SERPAPI_KEY)


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
            return "Web search is not available. Please rely on the article content and your knowledge."

        lines = [f"Search results for: {self.query}\n"]
        for i, r in enumerate(self.results[:5], 1):
            lines.append(f"{i}. {r.get('title', 'Untitled')}")
            if r.get("url"):
                lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                snippet = r["snippet"][:200]
                lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines)


async def _search_google_cse(query: str, num_results: int = 5) -> list[dict]:
    """Search using Google Custom Search API."""
    import aiohttp

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(num_results, 10),
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning("Google CSE error: %d", resp.status)
                return []
            data = await resp.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


async def _search_serpapi(query: str, num_results: int = 5) -> list[dict]:
    """Search using SerpAPI."""
    import aiohttp

    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_KEY,
        "q": query,
        "num": min(num_results, 10),
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning("SerpAPI error: %d", resp.status)
                return []
            data = await resp.json()

    results = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


async def web_search(query: str, num_results: int = 5) -> SearchResults:
    """Search the web for information.

    Uses Google Custom Search API or SerpAPI if configured.
    Returns empty results if no search backend is available.

    Args:
        query: Search query string
        num_results: Maximum number of results to return

    Returns:
        SearchResults with search results (or empty if unavailable)
    """
    if not SEARCH_ENABLED:
        logger.debug("Web search disabled (no API keys configured)")
        return SearchResults(query=query, results=[])

    try:
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            logger.debug("Web search (Google CSE): %s", query[:50])
            results = await _search_google_cse(query, num_results)
        elif SERPAPI_KEY:
            logger.debug("Web search (SerpAPI): %s", query[:50])
            results = await _search_serpapi(query, num_results)
        else:
            results = []

        logger.debug("Web search complete | query=%s results=%d", query[:50], len(results))
        return SearchResults(query=query, results=results)

    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return SearchResults(query=query, results=[])
