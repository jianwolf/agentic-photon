"""Async RSS feed parsing module.

This module handles concurrent fetching and parsing of RSS feeds.
It converts feed entries into Story objects for pipeline processing.

Features:
    - Concurrent fetching with connection pooling
    - SSL certificate handling with fallback
    - Age-based filtering of old stories
    - Graceful error handling per feed

Error Handling Strategy:
    - Individual feed failures don't affect other feeds
    - SSL errors trigger a retry without verification
    - Parse errors result in empty story list for that feed
    - All errors are logged at DEBUG level to avoid noise
"""

import asyncio
import logging
import ssl
from datetime import datetime, timedelta, timezone

import aiohttp
import certifi
import feedparser

from models.story import Story

logger = logging.getLogger(__name__)

# Browser-like User-Agent to avoid being blocked by some servers
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _ssl_context(verify: bool = True) -> ssl.SSLContext:
    """Create SSL context with optional certificate verification.

    Args:
        verify: If True, verify SSL certificates using certifi bundle.
                If False, disable verification (for problematic servers).

    Returns:
        Configured SSL context
    """
    if verify:
        return ssl.create_default_context(cafile=certifi.where())
    # Fallback: disable verification for servers with cert issues
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _parse_date(entry: dict) -> datetime | None:
    """Extract publication date from feed entry.

    Tries multiple date fields in order of preference:
    1. published_parsed - Standard RSS pubDate
    2. updated_parsed - Atom updated timestamp
    3. created_parsed - Less common creation date

    Args:
        entry: Parsed feed entry dictionary

    Returns:
        Datetime in UTC, or None if no valid date found
    """
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        time_tuple = entry.get(field)
        if time_tuple:
            try:
                return datetime(*time_tuple[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


async def _fetch_feed(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int,
    verify_ssl: bool = True,
) -> str | None:
    """Fetch feed content from URL with SSL fallback.

    On SSL certificate errors, automatically retries without verification.
    This handles servers with expired or self-signed certificates.

    Args:
        session: aiohttp client session
        url: Feed URL to fetch
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Feed content as string, or None on any error
    """
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": USER_AGENT},
            ssl=_ssl_context(verify_ssl),
        ) as resp:
            if resp.status != 200:
                if resp.status >= 500:
                    logger.warning("Feed %s: server error HTTP %d", url, resp.status)
                else:
                    logger.debug("Feed %s: HTTP %d", url, resp.status)
                return None
            return await resp.text()
    except aiohttp.ClientSSLError as e:
        # Retry without SSL verification on certificate errors
        if verify_ssl:
            logger.debug("Feed %s: SSL error, retrying without verification", url)
            return await _fetch_feed(session, url, timeout, verify_ssl=False)
        logger.warning("Feed %s: SSL verification failed after retry: %s", url, e)
        return None
    except asyncio.TimeoutError:
        logger.warning("Feed %s: request timed out after %ds", url, timeout)
        return None
    except Exception as e:
        logger.warning("Feed %s: %s: %s", url, type(e).__name__, e)
        return None


def _parse_feed_content(content: str, source_url: str, max_age_hours: int) -> list[Story]:
    """Parse feed content into Story objects.

    Filters out:
    - Entries without titles
    - Entries older than max_age_hours

    Args:
        content: Raw feed content (XML/RSS/Atom)
        source_url: URL of the feed (stored in Story.source_url)
        max_age_hours: Maximum age of stories to include

    Returns:
        List of Story objects (may be empty)
    """
    feed = feedparser.parse(content)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    stories = []

    for entry in feed.entries:
        # Skip entries without titles
        title = entry.get("title", "").strip()
        if not title:
            continue

        # Use current time if no publication date found (with warning)
        pub_date = _parse_date(entry)
        if not pub_date:
            pub_date = datetime.now(timezone.utc)
            logger.debug("Feed entry missing date, using current time: %s", title[:50])

        # Skip old entries
        if pub_date < cutoff:
            continue

        stories.append(Story(
            title=title,
            description=entry.get("description", "") or entry.get("summary", ""),
            pub_date=pub_date,
            source_url=source_url,
            article_url=entry.get("link", ""),
        ))

    return stories


async def _parse_feed(
    session: aiohttp.ClientSession,
    url: str,
    max_age_hours: int,
    timeout: int,
) -> list[Story]:
    """Fetch and parse a single feed.

    Combines fetching and parsing. Returns empty list on any error.

    Args:
        session: aiohttp client session
        url: Feed URL
        max_age_hours: Maximum story age to include
        timeout: Request timeout in seconds

    Returns:
        List of Story objects from the feed
    """
    content = await _fetch_feed(session, url, timeout)
    if not content:
        return []
    return _parse_feed_content(content, url, max_age_hours)


async def fetch_all_feeds(
    urls: list[str],
    max_age_hours: int,
    timeout: int = 30,
    max_concurrent: int = 10,
) -> list[Story]:
    """Fetch and parse all RSS feeds concurrently.

    Uses connection pooling to efficiently fetch multiple feeds in parallel.
    Individual feed failures are logged but don't affect other feeds.

    Args:
        urls: Feed URLs to fetch
        max_age_hours: Maximum story age to include (older filtered out)
        timeout: Request timeout per feed in seconds
        max_concurrent: Maximum concurrent TCP connections

    Returns:
        Combined list of Story objects from all successful feeds

    Example:
        >>> stories = await fetch_all_feeds(
        ...     urls=["https://example.com/feed.xml"],
        ...     max_age_hours=720,
        ...     timeout=30,
        ...     max_concurrent=10,
        ... )
        >>> len(stories)
        42
    """
    # Create connection pool with concurrency limit
    connector = aiohttp.TCPConnector(limit=max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Launch all feed fetches concurrently
        tasks = [_parse_feed(session, url, max_age_hours, timeout) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate results, tracking errors
    stories = []
    errors = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Feed error %s: %s (%s)", urls[i], result, type(result).__name__)
            errors += 1
        elif result:
            stories.extend(result)
            logger.debug("Feed %s: %d stories", urls[i], len(result))

    logger.info("Feeds fetched | stories=%d feeds=%d errors=%d", len(stories), len(urls), errors)
    return stories
