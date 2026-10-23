"""Logging setup.

Convention:
- Modules log via `logger = logging.getLogger(__name__)`; they never configure logging.
- CLI entry points call `setup_logging()` once in `main()`.
- Logs go to stderr; a CLI's actual result (answer, hits) goes to stdout.
"""

import logging

from agentic_now.config import settings

NOISY_LIBRARIES = ("httpx", "httpx2", "httpcore", "openai", "urllib3")


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(module)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
