"""Logging utilities with structured output and context propagation.

This module provides enhanced logging capabilities:
    - JSON structured logging for log aggregation systems
    - Run ID context propagation across all log messages
    - Consistent formatting and filtering

Usage:
    >>> from observability.logging import setup_logging, set_run_context
    >>> setup_logging(config)
    >>> set_run_context(run_id="abc123")
    >>> logger.info("Processing started")  # Includes run_id automatically
"""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from pathlib import Path
from typing import Any

# Context variable for run ID propagation
run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")

# Context variable for trace ID (Logfire integration)
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def set_run_context(run_id: str) -> None:
    """Set the current run ID for log context propagation.

    Args:
        run_id: Unique identifier for the current pipeline run
    """
    run_id_var.set(run_id)


def set_trace_context(trace_id: str) -> None:
    """Set the current trace ID for Logfire correlation.

    Args:
        trace_id: Trace ID from Logfire/OpenTelemetry
    """
    trace_id_var.set(trace_id)


def clear_context() -> None:
    """Clear all logging context variables."""
    run_id_var.set("-")
    trace_id_var.set("-")


class ContextFilter(logging.Filter):
    """Filter that injects context variables into log records.

    Adds run_id and trace_id to every log record for correlation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to the log record.

        Args:
            record: Log record to modify

        Returns:
            True (always passes the record through)
        """
        record.run_id = run_id_var.get()
        record.trace_id = trace_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs log records as single-line JSON objects suitable for
    log aggregation systems like ELK, Splunk, or Datadog.

    Output format:
        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string representation
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
        }

        # Add trace_id only if set (Logfire enabled)
        trace_id = getattr(record, "trace_id", "-")
        if trace_id != "-":
            log_data["trace_id"] = trace_id

        # Add source location for debugging
        if record.levelno >= logging.WARNING:
            log_data["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "run_id", "trace_id", "message",
            ):
                try:
                    json.dumps(value)  # Test serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter with context information.

    Format: TIMESTAMP [LEVEL] [run_id] logger: message
    """

    def __init__(self, include_date: bool = False):
        """Initialize the text formatter.

        Args:
            include_date: If True, include full date; otherwise just time
        """
        if include_date:
            datefmt = "%Y-%m-%d %H:%M:%S"
        else:
            datefmt = "%H:%M:%S"

        super().__init__(
            fmt="%(asctime)s [%(levelname)s] [%(run_id)s] %(name)s: %(message)s",
            datefmt=datefmt,
        )


def setup_logging(
    config: Any,
    verbose: bool = False,
) -> bool:
    """Configure logging with console and file handlers.

    Sets up dual logging to both console and rotating log files.
    Supports both text and JSON output formats.

    If the log directory is not writable, falls back to console-only logging.

    Args:
        config: Application configuration with logging settings
        verbose: If True, override config and use DEBUG level for console

    Returns:
        True if file logging is enabled, False if console-only (fallback)
    """
    # Determine log level
    if verbose:
        console_level = logging.DEBUG
    else:
        console_level = getattr(logging, config.log_level, logging.INFO)

    # Create context filter for run_id propagation
    context_filter = ContextFilter()

    # Create formatters based on config
    if config.log_format == "json":
        console_fmt = JsonFormatter()
        file_fmt = JsonFormatter()
    else:
        console_fmt = TextFormatter(include_date=False)
        file_fmt = TextFormatter(include_date=True)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(console_fmt)
    console.addFilter(context_filter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.addHandler(console)

    # Try to set up file handler with graceful degradation
    file_logging_enabled = False
    try:
        # Verify directory is writable
        config.log_dir.mkdir(parents=True, exist_ok=True)
        test_file = config.log_dir / ".write_test"
        test_file.touch()
        test_file.unlink()

        log_file = config.log_dir / "photon.log"

        # Choose rotation strategy based on config
        if config.log_max_bytes > 0:
            # Size-based rotation
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.log_max_bytes,
                backupCount=config.log_backup_count,
                encoding="utf-8",
            )
        else:
            # Time-based rotation (daily at midnight)
            file_handler = TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=1,
                backupCount=config.log_backup_count,
                encoding="utf-8",
            )

        file_handler.setLevel(logging.DEBUG)  # File always captures everything
        file_handler.setFormatter(file_fmt)
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)
        file_logging_enabled = True

    except (OSError, PermissionError) as e:
        # Log directory not writable - fall back to console only
        print(
            f"Warning: Cannot write to log directory '{config.log_dir}': {e}. "
            "Falling back to console-only logging.",
            file=sys.stderr
        )

    # Reduce noise from third-party libraries
    for lib in ("aiohttp", "urllib3", "httpx", "httpcore", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return file_logging_enabled


def get_logger(name: str) -> logging.Logger:
    """Get a logger with context support.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
