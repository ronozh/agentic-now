"""Observability: Pydantic AI spans + our own `@observe` spans -> Langfuse (OpenTelemetry).

Call `init_observability()` once per entry point. Without Langfuse keys, tracing is off
and the app runs unchanged; if Langfuse is down, exports fail in the background only.
"""

import logging
import os
from functools import cache

from langfuse import Langfuse, get_client
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


@cache
def init_observability() -> Langfuse | None:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logger.info("Langfuse keys not set; tracing disabled")
        return None
    Agent.instrument_all()  # every agent run, model request and tool call becomes a span
    client = get_client()
    logger.debug("Langfuse tracing to %s", os.getenv("LANGFUSE_BASE_URL"))
    return client


def flush() -> None:
    """Send buffered spans now; short-lived CLIs would otherwise exit before export."""
    if client := init_observability():
        client.flush()
