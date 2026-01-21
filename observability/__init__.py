"""Observability infrastructure for tracing and monitoring.

This package provides optional distributed tracing using Logfire.

setup_tracing:
    Initialize Logfire with PydanticAI instrumentation.

trace_operation:
    Context manager for custom span creation.

TracingContext:
    Configuration dataclass for tracing state.

Requirements:
    pip install logfire

Enable via configuration:
    ENABLE_LOGFIRE=true
    LOGFIRE_TOKEN=your-token  # Optional

Example:
    >>> from observability import setup_tracing, trace_operation
    >>> setup_tracing(enabled=True, service_name="photon")
    >>> with trace_operation("fetch_feeds"):
    ...     # Your code here
    ...     pass
"""

from observability.tracing import setup_tracing, trace_operation, TracingContext

__all__ = [
    "setup_tracing",
    "trace_operation",
    "TracingContext",
]
