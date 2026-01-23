"""Shared utilities for tools module.

This module contains shared constants and utility functions
used by multiple tools to avoid code duplication.
"""

import ssl
import certifi

# Browser-like User-Agent to avoid being blocked by some servers
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def create_ssl_context(verify: bool = True) -> ssl.SSLContext:
    """Create SSL context with optional certificate verification.

    Args:
        verify: If True, verify SSL certificates using certifi bundle.
                If False, disable verification (for problematic servers).

    Returns:
        Configured SSL context
    """
    if verify:
        return ssl.create_default_context(cafile=certifi.where())
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
