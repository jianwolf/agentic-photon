"""Configuration management for Photon news analysis pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    """Get string environment variable."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    val = os.environ.get(key)
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    val = os.environ.get(key)
    return float(val) if val else default


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


# Curated tech/AI RSS feeds
DEFAULT_RSS_URLS = [
    # Individual tech bloggers
    "https://karpathy.github.io/feed.xml",
    "https://simonwillison.net/atom/everything/",
    "https://jvns.ca/atom.xml",
    "https://danluu.com/atom.xml",
    "https://lilianweng.github.io/index.xml",
    "https://martinfowler.com/feed.atom",
    "https://newsletter.pragmaticengineer.com/feed",
    # AI/ML research labs
    "https://openai.com/blog/rss.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://thegradient.pub/rss/",
    "https://bair.berkeley.edu/blog/feed.xml",
    "https://huggingface.co/blog/feed.xml",
    # Tech company engineering blogs
    "https://netflixtechblog.com/feed",
    "https://stripe.com/blog/feed.rss",
    "https://engineering.atspotify.com/feed/",
    "https://blog.cloudflare.com/rss/",
    "https://tech.instacart.com/feed",
]


@dataclass
class Config:
    """Application configuration.

    All settings can be overridden via environment variables.
    """

    # Required
    gemini_api_key: str = ""

    # RSS sources
    rss_urls: list[str] = field(default_factory=lambda: DEFAULT_RSS_URLS.copy())

    # Output language: "zh" (Chinese) or "en" (English)
    language: str = "zh"

    # Models (PydanticAI format: provider:model)
    classifier_model: str = "google-gla:gemini-2.0-flash"
    researcher_model: str = "google-gla:gemini-2.0-flash"

    # Story filtering
    max_age_hours: int = 720  # 30 days

    # Database
    db_path: Path = field(default_factory=lambda: Path("news.db"))
    prune_after_days: int = 30

    # Pipeline
    poll_interval_seconds: int = 300  # 5 minutes
    max_workers: int = 8

    # Notifications
    webhook_url: str = ""
    alerts_file: str = ""

    # Retry behavior
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Output directories
    log_dir: Path = field(default_factory=lambda: Path("log"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))

    # Optional: Vector memory (requires chromadb)
    enable_memory: bool = False
    vector_db_path: Path = field(default_factory=lambda: Path("vectors"))

    # Optional: Observability (requires logfire)
    enable_logfire: bool = False
    logfire_token: str = ""

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            gemini_api_key=_env("GEMINI_API_KEY"),
            language=_env("LANGUAGE", "zh"),
            classifier_model=_env("CLASSIFIER_MODEL", "google-gla:gemini-2.0-flash"),
            researcher_model=_env("RESEARCHER_MODEL", "google-gla:gemini-2.0-flash"),
            max_age_hours=_env_int("MAX_AGE_HOURS", 720),
            db_path=Path(_env("DB_PATH", "news.db")),
            prune_after_days=_env_int("PRUNE_AFTER_DAYS", 30),
            poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 300),
            max_workers=_env_int("MAX_WORKERS", 8),
            webhook_url=_env("NOTIFICATION_WEBHOOK_URL"),
            alerts_file=_env("ALERTS_FILE"),
            max_retries=_env_int("MAX_RETRIES", 3),
            retry_base_delay=_env_float("RETRY_BASE_DELAY", 1.0),
            log_dir=Path(_env("LOG_DIR", "log")),
            reports_dir=Path(_env("REPORTS_DIR", "reports")),
            enable_memory=_env_bool("ENABLE_MEMORY", False),
            vector_db_path=Path(_env("VECTOR_DB_PATH", "vectors")),
            enable_logfire=_env_bool("ENABLE_LOGFIRE", False),
            logfire_token=_env("LOGFIRE_TOKEN"),
        )

    def validate(self) -> str | None:
        """Validate configuration.

        Returns:
            Error message string, or None if valid.
        """
        if not self.gemini_api_key:
            return "GEMINI_API_KEY environment variable is required"
        if not self.rss_urls:
            return "No RSS URLs configured"
        return None
