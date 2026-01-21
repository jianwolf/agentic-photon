"""Article fetching tool for retrieving full article content."""

import logging
import re
import ssl
from dataclasses import dataclass
from html.parser import HTMLParser
from io import StringIO

import aiohttp
import certifi

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class _HTMLTextExtractor(HTMLParser):
    """Extract text from HTML, skipping script/style tags."""

    SKIP_TAGS = frozenset({"script", "style", "head", "meta", "link"})

    def __init__(self):
        super().__init__()
        self._buffer = StringIO()
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._buffer.write(data)

    def get_text(self) -> str:
        return self._buffer.getvalue()


def _extract_text(html: str) -> str:
    """Extract readable text from HTML."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        text = parser.get_text()
    except Exception:
        # Fallback: strip tags with regex
        text = re.sub(r"<[^>]+>", " ", html)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ssl_context(verify: bool = True) -> ssl.SSLContext:
    """Create SSL context."""
    if verify:
        return ssl.create_default_context(cafile=certifi.where())
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


@dataclass
class ArticleContent:
    """Fetched article content."""

    url: str
    title: str
    content: str
    success: bool
    error: str | None = None

    @property
    def summary(self) -> str:
        """Get article summary for display."""
        if not self.success:
            return f"Failed to fetch: {self.error}"
        preview = self.content[:500] + "..." if len(self.content) > 500 else self.content
        return f"Title: {self.title}\n\n{preview}"


async def fetch_article(
    url: str,
    timeout: int = 30,
    max_length: int = 50000,
) -> ArticleContent:
    """Fetch and extract text from an article URL.

    Args:
        url: Article URL
        timeout: Request timeout in seconds
        max_length: Max content length

    Returns:
        ArticleContent with extracted text or error
    """
    logger.debug(f"Fetching: {url}")

    async def fetch_with_ssl(session: aiohttp.ClientSession, verify: bool) -> str:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": USER_AGENT},
            ssl=_ssl_context(verify),
        ) as resp:
            if resp.status != 200:
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=resp.status
                )
            return await resp.text()

    try:
        async with aiohttp.ClientSession() as session:
            try:
                html = await fetch_with_ssl(session, verify=True)
            except aiohttp.ClientSSLError:
                logger.debug(f"SSL error, retrying without verification: {url}")
                html = await fetch_with_ssl(session, verify=False)

        # Extract title
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = match.group(1).strip() if match else ""

        # Extract text
        content = _extract_text(html)
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"

        return ArticleContent(url=url, title=title, content=content, success=True)

    except aiohttp.ClientResponseError as e:
        return ArticleContent(url=url, title="", content="", success=False, error=f"HTTP {e.status}")
    except Exception as e:
        logger.error(f"Fetch error for {url}: {e}")
        return ArticleContent(url=url, title="", content="", success=False, error=str(e))
