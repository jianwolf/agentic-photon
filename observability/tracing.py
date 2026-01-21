"""Tracing and observability using Logfire/OpenTelemetry.

This module provides optional distributed tracing for the pipeline.
It integrates with Logfire (Pydantic's observability platform) and
automatically instruments PydanticAI agent calls.

Features:
    - Automatic PydanticAI instrumentation
    - Context manager for custom spans
    - Decorator for function tracing
    - Pipeline-level statistics tracking

Requirements:
    pip install logfire

Enable via configuration:
    ENABLE_LOGFIRE=true
    LOGFIRE_TOKEN=your-token  # Optional for cloud dashboard

Usage:
    >>> from observability.tracing import setup_tracing, trace_operation
    >>> setup_tracing(enabled=True, service_name="photon")
    >>> with trace_operation("my_operation"):
    ...     # Your code here
    ...     pass
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


@dataclass
class TracingContext:
    """Context for tracing operations."""
    enabled: bool = False
    service_name: str = "photon"
    token: str = ""
    _logfire_configured: bool = field(default=False, init=False)


_context = TracingContext()


def setup_tracing(
    enabled: bool = False,
    service_name: str = "photon",
    token: str = "",
) -> TracingContext:
    """Set up tracing with Logfire.

    Args:
        enabled: Whether to enable tracing
        service_name: Name of the service for tracing
        token: Logfire authentication token

    Returns:
        TracingContext for the session
    """
    global _context

    _context.enabled = enabled
    _context.service_name = service_name
    _context.token = token

    if not enabled:
        logger.debug("Tracing disabled")
        return _context

    try:
        import logfire

        logfire.configure(
            service_name=service_name,
            token=token if token else None,
        )

        # Instrument PydanticAI
        logfire.instrument_pydantic_ai()

        _context._logfire_configured = True
        logger.info(f"Logfire tracing enabled for service: {service_name}")

    except ImportError:
        logger.warning("Logfire not installed. Tracing disabled.")
        _context.enabled = False
    except Exception as e:
        logger.error(f"Failed to configure Logfire: {e}")
        _context.enabled = False

    return _context


@contextmanager
def trace_operation(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for tracing an operation.

    Args:
        name: Name of the operation
        attributes: Optional attributes to attach to the span

    Yields:
        Dictionary for adding additional attributes during the operation
    """
    span_attrs = attributes or {}
    start_time = datetime.now()

    try:
        if _context.enabled and _context._logfire_configured:
            import logfire

            with logfire.span(name, **span_attrs) as span:
                result_attrs: dict[str, Any] = {}
                yield result_attrs

                # Add any result attributes to the span
                for key, value in result_attrs.items():
                    span.set_attribute(key, value)
        else:
            # No-op context manager when tracing is disabled
            result_attrs = {}
            yield result_attrs

    finally:
        duration = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Operation '{name}' completed in {duration:.2f}s")


def trace_function(name: str | None = None) -> Callable:
    """Decorator for tracing a function.

    Args:
        name: Optional name for the span (defaults to function name)

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_operation(span_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_operation(span_name):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class PipelineTracer:
    """Tracer for the news analysis pipeline."""

    def __init__(self, context: TracingContext | None = None):
        """Initialize the pipeline tracer.

        Args:
            context: Optional tracing context (uses global if not provided)
        """
        self.context = context or _context
        self.run_id: str | None = None
        self.start_time: datetime | None = None
        self.stats: dict[str, Any] = {}

    @contextmanager
    def trace_run(self, run_id: str) -> Generator[None, None, None]:
        """Trace a complete pipeline run.

        Args:
            run_id: Unique identifier for this run
        """
        self.run_id = run_id
        self.start_time = datetime.now()
        self.stats = {
            "run_id": run_id,
            "start_time": self.start_time.isoformat(),
        }

        with trace_operation("pipeline_run", {"run_id": run_id}) as attrs:
            try:
                yield
            finally:
                duration = (datetime.now() - self.start_time).total_seconds()
                self.stats["duration_seconds"] = duration
                attrs.update(self.stats)

    def record_fetch(self, feed_count: int, story_count: int) -> None:
        """Record feed fetch statistics.

        Args:
            feed_count: Number of feeds fetched
            story_count: Number of stories found
        """
        self.stats["feeds_fetched"] = feed_count
        self.stats["stories_fetched"] = story_count
        logger.info(f"Fetched {story_count} stories from {feed_count} feeds")

    def record_dedup(self, before: int, after: int) -> None:
        """Record deduplication statistics.

        Args:
            before: Stories before dedup
            after: Stories after dedup
        """
        self.stats["stories_before_dedup"] = before
        self.stats["stories_after_dedup"] = after
        self.stats["stories_deduplicated"] = before - after
        logger.info(f"Dedup: {before} -> {after} stories ({before - after} removed)")

    def record_classification(
        self,
        total: int,
        important: int,
        not_important: int,
    ) -> None:
        """Record classification statistics.

        Args:
            total: Total stories classified
            important: Stories marked important
            not_important: Stories marked not important
        """
        self.stats["stories_classified"] = total
        self.stats["stories_important"] = important
        self.stats["stories_not_important"] = not_important
        logger.info(f"Classified: {important} important, {not_important} not important")

    def record_analysis(self, analyzed: int, errors: int) -> None:
        """Record analysis statistics.

        Args:
            analyzed: Stories successfully analyzed
            errors: Analysis errors
        """
        self.stats["stories_analyzed"] = analyzed
        self.stats["analysis_errors"] = errors
        logger.info(f"Analyzed: {analyzed} stories, {errors} errors")

    def record_notification(self, notified: int, failed: int) -> None:
        """Record notification statistics.

        Args:
            notified: Successful notifications
            failed: Failed notifications
        """
        self.stats["notifications_sent"] = notified
        self.stats["notifications_failed"] = failed
        logger.info(f"Notified: {notified} sent, {failed} failed")

    def get_summary(self) -> dict[str, Any]:
        """Get the complete run summary.

        Returns:
            Dictionary of all recorded statistics
        """
        return self.stats.copy()
