"""Async RSS feed parsing module."""

import asyncio
import logging
import ssl
from datetime import datetime, timedelta, timezone

import aiohttp
import certifi
import feedparser

from models.story import Story

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _ssl_context(verify: bool = True) -> ssl.SSLContext:
    """Create SSL context with optional verification."""
    if verify:
        return ssl.create_default_context(cafile=certifi.where())
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _parse_date(entry: dict) -> datetime | None:
    """Extract publication date from feed entry."""
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
    """Fetch feed content from URL."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": USER_AGENT},
            ssl=_ssl_context(verify_ssl),
        ) as resp:
            if resp.status != 200:
                logger.debug(f"Feed {url}: HTTP {resp.status}")
                return None
            return await resp.text()
    except aiohttp.ClientSSLError:
        if verify_ssl:
            logger.debug(f"Feed {url}: SSL error, retrying")
            return await _fetch_feed(session, url, timeout, verify_ssl=False)
        return None
    except asyncio.TimeoutError:
        logger.debug(f"Feed {url}: timeout")
        return None
    except Exception as e:
        logger.debug(f"Feed {url}: {e}")
        return None


def _parse_feed_content(content: str, source_url: str, max_age_hours: int) -> list[Story]:
    """Parse feed content into Story objects."""
    feed = feedparser.parse(content)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    stories = []

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        pub_date = _parse_date(entry) or datetime.now(timezone.utc)
        if pub_date < cutoff:
            continue

        stories.append(Story(
            title=title,
            description=entry.get("description", "") or entry.get("summary", ""),
            pub_date=pub_date,
            source_url=source_url,
        ))

    return stories


async def _parse_feed(
    session: aiohttp.ClientSession,
    url: str,
    max_age_hours: int,
    timeout: int,
) -> list[Story]:
    """Fetch and parse a single feed."""
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

    Args:
        urls: Feed URLs to fetch
        max_age_hours: Max story age to include
        timeout: Request timeout per feed
        max_concurrent: Max concurrent requests

    Returns:
        Combined list of Story objects
    """
    connector = aiohttp.TCPConnector(limit=max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_parse_feed(session, url, max_age_hours, timeout) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    stories = []
    errors = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.debug(f"Feed error {urls[i]}: {result}")
            errors += 1
        elif result:
            stories.extend(result)

    logger.info(f"Feeds: {len(stories)} stories from {len(urls)} feeds ({errors} errors)")
    return stories
