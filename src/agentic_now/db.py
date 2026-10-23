import psycopg
from pgvector.psycopg import register_vector

from agentic_now.config import settings


def connect(**kwargs) -> psycopg.Connection:
    """Connection with the pgvector type registered."""
    conn = psycopg.connect(settings.database_url, **kwargs)
    register_vector(conn)
    return conn


def init_extensions() -> str:
    """Enable pgvector; return its version."""
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        return conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]


if __name__ == "__main__":
    print(f"pgvector {init_extensions()}")
