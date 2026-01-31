"""Article fetching tool for retrieving full article content.

This module provides the article fetching tool used by the researcher agent.
It fetches HTML from URLs and extracts readable text content.

Features:
    - SSL fallback for problematic certificates
    - HTML-to-text conversion (strips scripts, styles)
    - Content truncation for large pages
    - Error handling with informative messages

The extracted text is used by the researcher agent to understand
the full context of news stories beyond the RSS description.
"""

import gzip
import html
import logging
import re
import zlib
from dataclasses import dataclass
from html.parser import HTMLParser
from io import StringIO

import aiohttp

from tools.utils import create_ssl_context, USER_AGENT

logger = logging.getLogger(__name__)
try:
    import brotli  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    brotli = None


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML, skipping non-content tags.

    Ignores content within script, style, head, meta, and link tags.
    Accumulates all other text content into a buffer.

    Usage:
        >>> parser = _HTMLTextExtractor()
        >>> parser.feed("<p>Hello <script>ignored</script> world</p>")
        >>> parser.get_text()
        'Hello  world'
    """

    # Tags whose content should be completely ignored
    SKIP_TAGS = frozenset({"script", "style", "head", "meta", "link"})

    def __init__(self):
        super().__init__()
        self._buffer = StringIO()
        self._skip_depth = 0  # Nesting depth within skip tags

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        # Only capture text when not inside a skip tag
        if self._skip_depth == 0:
            self._buffer.write(data)

    def get_text(self) -> str:
        """Return accumulated text content."""
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
    logger.debug("Fetching article: %s", url)

    def _decode_body(resp: aiohttp.ClientResponse, body: bytes) -> str:
        enc_header = resp.headers.get("Content-Encoding", "")
        encodings = [enc.strip().lower() for enc in enc_header.split(",") if enc.strip()]
        for encoding in reversed(encodings):
            if encoding == "br":
                if brotli is None:
                    raise RuntimeError("brotli not available to decode br content")
                body = brotli.decompress(body)
            elif encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding == "deflate":
                try:
                    body = zlib.decompress(body)
                except zlib.error:
                    body = zlib.decompress(body, -zlib.MAX_WBITS)
            elif encoding in ("identity", ""):
                continue
            else:
                raise RuntimeError(f"Unsupported content encoding: {encoding}")
        charset = resp.charset or "utf-8"
        return body.decode(charset, errors="replace")

    async def fetch_with_ssl(session: aiohttp.ClientSession, verify: bool) -> str:
        accept_encoding = "gzip, deflate, br" if brotli is not None else "gzip, deflate"
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": accept_encoding},
            auto_decompress=False,
            ssl=create_ssl_context(verify),
        ) as resp:
            if resp.status != 200:
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=resp.status
                )
            raw = await resp.read()
            return _decode_body(resp, raw)

    try:
        async with aiohttp.ClientSession(auto_decompress=False) as session:
            try:
                html_content = await fetch_with_ssl(session, verify=True)
            except aiohttp.ClientSSLError:
                logger.debug("SSL error, retrying without verification: %s", url)
                html_content = await fetch_with_ssl(session, verify=False)

        # Extract title and decode HTML entities
        match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
        title = html.unescape(match.group(1).strip()) if match else ""

        # Extract text
        content = _extract_text(html_content)
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"

        return ArticleContent(url=url, title=title, content=content, success=True)

    except aiohttp.ClientResponseError as e:
        return ArticleContent(url=url, title="", content="", success=False, error=f"HTTP {e.status}")
    except Exception as e:
        logger.error("Fetch error for %s: %s", url, e, exc_info=True)
        return ArticleContent(url=url, title="", content="", success=False, error=str(e))
