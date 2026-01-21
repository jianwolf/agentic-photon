"""Story data model for RSS feed items."""

from datetime import datetime
from hashlib import sha256
import re
import unicodedata

from pydantic import BaseModel, Field


class Story(BaseModel):
    """A news story fetched from an RSS feed.

    Attributes:
        title: Article headline
        description: Raw HTML/text content from RSS feed entry
        pub_date: Publication timestamp (UTC)
        source_url: URL of the RSS feed this story came from
    """

    title: str = Field(description="Article headline")
    description: str = Field(default="", description="HTML/text from RSS entry")
    pub_date: datetime = Field(description="Publication timestamp")
    source_url: str = Field(description="RSS feed URL")

    @property
    def hash(self) -> str:
        """Generate a 16-character deduplication hash.

        Combines normalized title, publication hour, and source URL.
        Stories with the same title from the same source in the same
        hour will have the same hash.
        """
        normalized = unicodedata.normalize("NFKC", self.title.lower())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        hour_str = self.pub_date.strftime("%Y-%m-%d-%H")
        payload = f"{normalized}|{hour_str}|{self.source_url}"

        return sha256(payload.encode()).hexdigest()[:16]

    @property
    def publishers(self) -> list[str]:
        """Extract publisher names from HTML description.

        Looks for patterns like "via <a>Publisher</a>" common in
        aggregator feeds.

        Returns:
            List of unique publisher names found, or empty list.
        """
        if not self.description:
            return []

        patterns = [
            r'via\s+<a[^>]*>([^<]+)</a>',
            r'from\s+<a[^>]*>([^<]+)</a>',
            r'by\s+<a[^>]*>([^<]+)</a>',
        ]

        publishers = []
        for pattern in patterns:
            matches = re.findall(pattern, self.description, re.IGNORECASE)
            publishers.extend(matches)

        return list(set(publishers))
