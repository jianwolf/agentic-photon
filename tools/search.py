"""Web search tool for fact-checking and research.

This module provides the web search tool used by the researcher agent.

CURRENT STATUS: Search is disabled (returns empty results).
The researcher agent will rely on article fetching and its training data.

To enable real search, implement one of these backends:
    1. Google Custom Search API (set GOOGLE_CSE_ID and GOOGLE_API_KEY)
    2. SerpAPI (set SERPAPI_KEY)

NOTE: Do NOT use Gemini with search grounding here - that would create
double LLM calls (researcher agent + search tool), wasting tokens.
Use a direct search API instead.

Error Handling:
    - Search disabled: Returns "not available" message (soft)
    - Search enabled but API error: Raises SearchError (hard)
    - Search enabled but no results: Returns empty results (soft - query may be bad)
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


class SearchError(Exception):
    """Raised when search API fails with a non-recoverable error.

    This is a hard error that should not be retried. Examples:
    - Invalid API key
    - Quota exceeded
    - API returned error status
    """
    pass


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
    def is_enabled(self) -> bool:
        """Check if search was attempted (API configured)."""
        return SEARCH_ENABLED

    @property
    def summary(self) -> str:
        """Format results as a readable summary for the agent.

        Returns:
            Human-readable string with numbered results
        """
        if not self.results:
            if not self.is_enabled:
                return "Web search is not configured. Please rely on the article content and your knowledge."
            return f"No search results found for: {self.query}"

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
    """Search using Google Custom Search API.

    Raises:
        SearchError: If API returns error status
    """
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
            if resp.status == 403:
                raise SearchError("Google CSE API key invalid or quota exceeded")
            if resp.status != 200:
                raise SearchError(f"Google CSE API error: HTTP {resp.status}")
            data = await resp.json()

    if "error" in data:
        raise SearchError(f"Google CSE error: {data['error'].get('message', 'Unknown')}")

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


async def _search_serpapi(query: str, num_results: int = 5) -> list[dict]:
    """Search using SerpAPI.

    Raises:
        SearchError: If API returns error status
    """
    import aiohttp

    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_KEY,
        "q": query,
        "num": min(num_results, 10),
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 401:
                raise SearchError("SerpAPI key invalid")
            if resp.status == 429:
                raise SearchError("SerpAPI quota exceeded")
            if resp.status != 200:
                raise SearchError(f"SerpAPI error: HTTP {resp.status}")
            data = await resp.json()

    if "error" in data:
        raise SearchError(f"SerpAPI error: {data['error']}")

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

    Raises:
        SearchError: If search is enabled but API fails (hard error, don't retry)
    """
    if not SEARCH_ENABLED:
        logger.debug("Web search disabled (no API keys configured)")
        return SearchResults(query=query, results=[])

    # Search is enabled - errors should be hard failures
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
