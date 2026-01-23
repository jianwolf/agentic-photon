"""Tools for the Photon news analysis agents.

This package provides tools used by the researcher agent:

web_search (placeholder):
    Search the web for fact-checking and background.
    Currently returns empty results - see search.py for integration guide.

fetch_article:
    Fetch and extract text from article URLs.
    Handles SSL fallback and HTML parsing.

query_related_stories:
    Query database for previously analyzed stories.
    Uses keyword matching on title and summary.

Each tool returns a result object with a .summary property
formatted for agent consumption.

Example:
    >>> from tools import fetch_article, query_related_stories
    >>> content = await fetch_article("https://example.com/article")
    >>> print(content.summary)
"""

from tools.utils import create_ssl_context, USER_AGENT
from tools.search import web_search, SearchResults
from tools.fetch import fetch_article, ArticleContent
from tools.database import query_related_stories, StoryHistory

__all__ = [
    "web_search",
    "SearchResults",
    "fetch_article",
    "ArticleContent",
    "query_related_stories",
    "StoryHistory",
    "create_ssl_context",
    "USER_AGENT",
]
