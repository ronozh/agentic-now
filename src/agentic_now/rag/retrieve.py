"""Iteration 01 retrieval: embed the question, return the nearest chunks.

Run: uv run python -m agentic_now.rag.retrieve "question" [k]
"""

import logging
import sys
import time
from dataclasses import dataclass

import psycopg
from langfuse import get_client, observe
from pydantic import BaseModel

from agentic_now import db
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from agentic_now.rag.embed import embed_query

logger = logging.getLogger(__name__)


class Source(BaseModel, frozen=True):
    """A citable location in a book (frozen = hashable, so sources can be de-duplicated)."""

    title: str
    page: int  # PDF page
    printed_page: int | None = None

    @property
    def label(self) -> str:
        if self.printed_page:
            return f"{self.title}, p. {self.printed_page} (PDF p. {self.page})"
        return f"{self.title}, PDF p. {self.page}"


@dataclass
class Hit:
    title: str
    page: int
    printed_page: int | None
    content: str
    distance: float  # cosine distance: 0 = identical, 1 = unrelated

    @property
    def source(self) -> Source:
        return Source(title=self.title, page=self.page, printed_page=self.printed_page)


@observe(name="retrieve", as_type="retriever", capture_input=False)
def retrieve(
    conn: psycopg.Connection, query: str, k: int = 5, book: str | None = None
) -> list[Hit]:
    """Nearest chunks to the query; `book` optionally restricts to titles containing it."""
    get_client().update_current_span(input={"query": query, "k": k, "book": book})
    started = time.perf_counter()
    rows = conn.execute(
        """
        SELECT d.title, c.page, c.printed_page, c.content, c.embedding <=> %(q)s AS distance
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE %(book)s::text IS NULL OR d.title ILIKE '%%' || %(book)s || '%%'
        ORDER BY distance
        LIMIT %(k)s
        """,
        {"q": embed_query(query), "k": k, "book": book},
    ).fetchall()
    hits = [Hit(*row) for row in rows]
    logger.info(
        "retrieved %d hits in %.2fs (best distance %.3f) for %r",
        len(hits),
        time.perf_counter() - started,
        hits[0].distance if hits else float("nan"),
        query,
    )
    return hits


def main() -> None:
    setup_logging()
    init_observability()
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    with db.connect() as conn:
        for rank, hit in enumerate(retrieve(conn, query, k), start=1):
            print(f"[{rank}] {hit.distance:.3f}  {hit.source.label}")
            print(f"    {hit.content[:200]}…\n")
    flush()


if __name__ == "__main__":
    main()
