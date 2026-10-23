import logging
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from agentic_now.config import settings
from agentic_now.log import setup_logging

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


CONNECT_ATTEMPTS = 8
CONNECT_BACKOFF_SECONDS = 3


def connect(attempts: int = CONNECT_ATTEMPTS, **kwargs) -> psycopg.Connection:
    """Connection with the pgvector type registered.

    Retries while Postgres restarts or runs crash recovery (seen under heavy host memory
    pressure, see .plan/gotcha.md), so a long agent run survives a short outage. Probes such
    as /health pass attempts=1 to fail fast.
    """
    for attempt in range(1, attempts + 1):
        try:
            conn = psycopg.connect(settings.database_url, **kwargs)
            break
        except psycopg.OperationalError as e:
            if attempt == attempts:
                raise
            logger.warning(
                "Postgres not available (%s); retry %d", str(e).splitlines()[0], attempt
            )
            time.sleep(CONNECT_BACKOFF_SECONDS * attempt)
    register_vector(conn)
    return conn


def init_db() -> str:
    """Create extensions and tables (idempotent); return the pgvector version."""
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute((SQL_DIR / "rag_schema.sql").read_text())
        return conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]


if __name__ == "__main__":
    setup_logging()
    logger.info("schema ready, pgvector %s", init_db())
