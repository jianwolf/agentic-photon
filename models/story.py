"""Story data model for RSS feed items.

This module defines the core Story model used throughout the pipeline.
Each Story represents a single news item from an RSS feed.

Deduplication Strategy:
    Stories are deduplicated using a hash computed from:
    - Normalized title (lowercase, NFKC normalized, punctuation removed)
    - Publication hour (not full timestamp, to handle minor variations)
    - Source feed URL

    This allows the same story from different feeds to be distinct,
    while catching duplicates from the same source.
"""

from datetime import datetime
from hashlib import sha256
import re
import unicodedata

from pydantic import BaseModel, Field


class Story(BaseModel):
    """A news story fetched from an RSS feed.

    This is the primary data model for news items flowing through the pipeline.
    Stories are parsed from RSS feeds and processed by the classifier and
    researcher agents.

    Attributes:
        title: Article headline (required)
        description: Raw HTML/text content from RSS feed entry
        pub_date: Publication timestamp in UTC
        source_url: URL of the RSS feed this story came from

    Example:
        >>> story = Story(
        ...     title="OpenAI Announces GPT-5",
        ...     description="<p>Major breakthrough in AI...</p>",
        ...     pub_date=datetime.now(timezone.utc),
        ...     source_url="https://openai.com/blog/rss.xml"
        ... )
        >>> story.hash  # 16-char dedup key
        'a1b2c3d4e5f6g7h8'
    """

    title: str = Field(description="Article headline")
    description: str = Field(default="", description="HTML/text from RSS entry")
    pub_date: datetime = Field(description="Publication timestamp (UTC)")
    source_url: str = Field(description="URL of the source RSS feed")
    article_url: str = Field(default="", description="URL of the actual article")

    @property
    def hash(self) -> str:
        """Generate a 16-character deduplication hash.

        The hash combines:
        - Normalized title (lowercase, NFKC, no punctuation, single spaces)
        - Publication hour (YYYY-MM-DD-HH format)
        - Source feed URL

        This design ensures:
        - Same story from same source = same hash (deduped)
        - Same story from different sources = different hash (kept)
        - Minor title variations (punctuation) = same hash (deduped)
        - Different hours = different hash (allows republished stories)

        Returns:
            16-character hex string (first 16 chars of SHA-256)
        """
        # Normalize: lowercase, Unicode normalization, remove punctuation
        normalized = unicodedata.normalize("NFKC", self.title.lower())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Use hour granularity (not minute/second) for publication time
        hour_str = self.pub_date.strftime("%Y-%m-%d-%H")

        # Combine components with separator
        payload = f"{normalized}|{hour_str}|{self.source_url}"

        return sha256(payload.encode()).hexdigest()[:16]

    @property
    def publishers(self) -> list[str]:
        """Extract publisher names from HTML description.

        Many RSS aggregators include the original publisher in the description
        using patterns like "via <a href='...'>Publisher Name</a>".

        Recognized patterns:
        - "via <a>Publisher</a>"
        - "from <a>Publisher</a>"
        - "by <a>Publisher</a>"

        Returns:
            List of unique publisher names found, or empty list if none.
        """
        if not self.description:
            return []

        # Patterns for common aggregator formats
        patterns = [
            r'via\s+<a[^>]*>([^<]+)</a>',   # "via <a>Name</a>"
            r'from\s+<a[^>]*>([^<]+)</a>',  # "from <a>Name</a>"
            r'by\s+<a[^>]*>([^<]+)</a>',    # "by <a>Name</a>"
        ]

        publishers = []
        for pattern in patterns:
            matches = re.findall(pattern, self.description, re.IGNORECASE)
            publishers.extend(matches)

        # Return unique publishers preserving order
        return list(dict.fromkeys(publishers))

    def __str__(self) -> str:
        """Human-readable representation for logging."""
        return f"Story({self.hash[:8]}..., '{self.title[:50]}...')"
