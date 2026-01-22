"""Configuration management for Photon news analysis pipeline.

This module provides centralized configuration for all pipeline components.
All settings are loaded from environment variables with sensible defaults.

Environment Variables:
    Required:
        GEMINI_API_KEY: Google Gemini API key for AI agents

    Models (PydanticAI format - provider:model):
        CLASSIFIER_MODEL: Model for quick importance classification
        RESEARCHER_MODEL: Model for deep analysis with tools

    Output:
        LANGUAGE: Output language ('zh' for Chinese, 'en' for English)
        DB_PATH: SQLite database file path
        REPORTS_DIR: Directory for markdown reports
        LOG_DIR: Directory for log files

    Pipeline Behavior:
        MAX_AGE_HOURS: Maximum story age to process (default: 720 = 30 days)
        POLL_INTERVAL_SECONDS: Delay between runs in continuous mode
        MAX_WORKERS: Maximum concurrent operations

    Notifications:
        NOTIFICATION_WEBHOOK_URL: HTTP endpoint for story alerts
        ALERTS_FILE: Path for JSONL alert file

    Optional Features:
        ENABLE_MEMORY: Enable ChromaDB vector store for semantic search
        ENABLE_LOGFIRE: Enable Logfire/OpenTelemetry tracing

    Logging:
        LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_BACKUP_COUNT: Number of rotated log files to keep
        LOG_MAX_BYTES: Max log file size in bytes (0 = time-based rotation)
        LOG_FORMAT: Log format ('text' or 'json' for structured logging)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    """Get string environment variable with optional default.

    Args:
        key: Environment variable name
        default: Value to return if not set

    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Get integer environment variable with default.

    Args:
        key: Environment variable name
        default: Value to return if not set or invalid

    Returns:
        Parsed integer or default value
    """
    val = os.environ.get(key)
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    """Get float environment variable with default.

    Args:
        key: Environment variable name
        default: Value to return if not set or invalid

    Returns:
        Parsed float or default value
    """
    val = os.environ.get(key)
    return float(val) if val else default


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable with default.

    Recognizes truthy values: '1', 'true', 'yes', 'on'
    Recognizes falsy values: '0', 'false', 'no', 'off'

    Args:
        key: Environment variable name
        default: Value to return if not set or unrecognized

    Returns:
        Parsed boolean or default value
    """
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


# Curated tech/AI RSS feeds organized by category
# These are high-quality sources focused on technology, AI/ML, and software engineering
DEFAULT_RSS_URLS = [
    # === Individual Tech Bloggers ===
    # Influential voices in ML, systems, and software engineering
    "https://karpathy.github.io/feed.xml",       # Andrej Karpathy - ML/AI
    "https://simonwillison.net/atom/everything/", # Simon Willison - LLMs, data
    "https://jvns.ca/atom.xml",                   # Julia Evans - systems, debugging
    "https://danluu.com/atom.xml",                # Dan Luu - systems, performance
    "https://lilianweng.github.io/index.xml",     # Lilian Weng - ML research
    "https://martinfowler.com/feed.atom",         # Martin Fowler - architecture
    "https://newsletter.pragmaticengineer.com/feed",  # Gergely Orosz - eng culture

    # === AI/ML Research Labs ===
    # Official blogs from leading AI research organizations
    "https://openai.com/blog/rss.xml",            # OpenAI
    "https://deepmind.google/blog/rss.xml",       # Google DeepMind
    "https://thegradient.pub/rss/",               # The Gradient (ML publication)
    "https://bair.berkeley.edu/blog/feed.xml",    # Berkeley AI Research
    "https://huggingface.co/blog/feed.xml",       # Hugging Face

    # === Tech Company Engineering Blogs ===
    # Infrastructure, scale, and engineering practices
    "https://netflixtechblog.com/feed",           # Netflix
    "https://stripe.com/blog/feed.rss",           # Stripe
    "https://engineering.atspotify.com/feed/",    # Spotify
    "https://blog.cloudflare.com/rss/",           # Cloudflare
    "https://tech.instacart.com/feed",            # Instacart
]


@dataclass
class Config:
    """Application configuration loaded from environment variables.

    All settings can be overridden via environment variables. Use Config.load()
    to create an instance with values from the environment.

    Example:
        >>> config = Config.load()
        >>> if error := config.validate():
        ...     print(f"Config error: {error}")
    """

    # === Required ===
    gemini_api_key: str = ""  # GEMINI_API_KEY - Google AI API key

    # === RSS Sources ===
    rss_urls: list[str] = field(default_factory=lambda: DEFAULT_RSS_URLS.copy())

    # === Output Settings ===
    language: str = "zh"  # LANGUAGE - 'zh' (Chinese) or 'en' (English)

    # === AI Models ===
    # PydanticAI format: provider:model (e.g., 'google-gla:gemini-3-flash-preview')
    classifier_model: str = "google-gla:gemini-3-flash-preview"  # Fast classification
    researcher_model: str = "google-gla:gemini-3-flash-preview"  # Deep analysis

    # === Story Filtering ===
    max_age_hours: int = 720  # MAX_AGE_HOURS - Skip stories older than this (30 days)

    # === Database ===
    db_path: Path = field(default_factory=lambda: Path("news.db"))  # DB_PATH
    prune_after_days: int = 30  # PRUNE_AFTER_DAYS - Auto-delete older records

    # === Pipeline Behavior ===
    poll_interval_seconds: int = 300  # POLL_INTERVAL_SECONDS - Delay between runs
    max_workers: int = 8  # MAX_WORKERS - Concurrent feed fetches / API calls

    # === Notifications ===
    webhook_url: str = ""  # NOTIFICATION_WEBHOOK_URL - POST endpoint for alerts
    alerts_file: str = ""  # ALERTS_FILE - JSONL file path for alerts

    # === Retry Behavior ===
    max_retries: int = 3  # MAX_RETRIES - API call retry attempts
    retry_base_delay: float = 1.0  # RETRY_BASE_DELAY - Base delay for exponential backoff

    # === Output Directories ===
    log_dir: Path = field(default_factory=lambda: Path("log"))  # LOG_DIR
    reports_dir: Path = field(default_factory=lambda: Path("reports"))  # REPORTS_DIR

    # === Logging Configuration ===
    log_level: str = "INFO"  # LOG_LEVEL - DEBUG, INFO, WARNING, ERROR
    log_backup_count: int = 30  # LOG_BACKUP_COUNT - Number of rotated logs to keep
    log_max_bytes: int = 0  # LOG_MAX_BYTES - Max file size (0 = time-based rotation)
    log_format: str = "text"  # LOG_FORMAT - 'text' or 'json' for structured logging

    # === Optional: Vector Memory ===
    # Requires: pip install chromadb
    enable_memory: bool = False  # ENABLE_MEMORY - Enable semantic story search
    vector_db_path: Path = field(default_factory=lambda: Path("vectors"))

    # === Optional: Observability ===
    # Requires: pip install logfire
    enable_logfire: bool = False  # ENABLE_LOGFIRE - Enable distributed tracing
    logfire_token: str = ""  # LOGFIRE_TOKEN - Authentication token

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            gemini_api_key=_env("GEMINI_API_KEY"),
            language=_env("LANGUAGE", "zh"),
            classifier_model=_env("CLASSIFIER_MODEL", "google-gla:gemini-3-flash-preview"),
            researcher_model=_env("RESEARCHER_MODEL", "google-gla:gemini-3-flash-preview"),
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
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            log_backup_count=_env_int("LOG_BACKUP_COUNT", 30),
            log_max_bytes=_env_int("LOG_MAX_BYTES", 0),
            log_format=_env("LOG_FORMAT", "text").lower(),
        )

    def validate(self) -> str | None:
        """Validate configuration for required fields and valid values.

        Checks:
            - GEMINI_API_KEY is set
            - At least one RSS URL is configured
            - Language is 'zh' or 'en'
            - Numeric values are positive

        Returns:
            Error message string if invalid, None if valid.
        """
        if not self.gemini_api_key:
            return "GEMINI_API_KEY environment variable is required"
        if not self.rss_urls:
            return "No RSS URLs configured"
        if self.language not in ("zh", "en"):
            return f"Invalid LANGUAGE '{self.language}' - must be 'zh' or 'en'"
        if self.max_age_hours <= 0:
            return "MAX_AGE_HOURS must be positive"
        if self.poll_interval_seconds <= 0:
            return "POLL_INTERVAL_SECONDS must be positive"
        if self.max_workers <= 0:
            return "MAX_WORKERS must be positive"
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            return f"Invalid LOG_LEVEL '{self.log_level}' - must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
        if self.log_format not in ("text", "json"):
            return f"Invalid LOG_FORMAT '{self.log_format}' - must be 'text' or 'json'"
        if self.log_backup_count < 0:
            return "LOG_BACKUP_COUNT must be non-negative"
        return None
