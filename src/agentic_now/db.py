import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from agentic_now.config import settings
from agentic_now.log import setup_logging

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def connect(**kwargs) -> psycopg.Connection:
    """Connection with the pgvector type registered."""
    conn = psycopg.connect(settings.database_url, **kwargs)
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
