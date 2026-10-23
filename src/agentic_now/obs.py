"""Observability: Pydantic AI spans + our own `@observe` spans -> Langfuse (OpenTelemetry).

Call `init_observability()` once per entry point. Tracing is off (and the app runs unchanged)
without Langfuse keys or when Langfuse isn't reachable, so it can stay stopped to save memory.
"""

import logging
import os
import urllib.request
from functools import cache

from langfuse import Langfuse, get_client
from pydantic_ai import Agent

from agentic_now.config import settings

logger = logging.getLogger(__name__)


def _langfuse_reachable() -> bool:
    try:
        url = f"{settings.langfuse_base_url}/api/public/health"
        with urllib.request.urlopen(url, timeout=2):
            return True
    except OSError:
        return False


@cache
def init_observability() -> Langfuse | None:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        reason = "Langfuse keys not set"
    elif not _langfuse_reachable():
        reason = "Langfuse not reachable"
    else:
        reason = None
    if reason:
        # Without this, the SDK would still buffer spans and retry exports to a dead server.
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logger.info("%s; tracing disabled", reason)
        return None
    Agent.instrument_all()  # every agent run, model request and tool call becomes a span
    client = get_client()
    logger.debug("Langfuse tracing to %s", settings.langfuse_base_url)
    return client


def flush() -> None:
    """Send buffered spans now; short-lived CLIs would otherwise exit before export."""
    if client := init_observability():
        client.flush()
