"""Iteration 01 ingestion: PDF -> pages -> chunks -> embeddings -> pgvector.

Run: uv run python -m agentic_now.rag.ingest data/books/        (a folder or one .pdf)
"""

import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pymupdf
from langfuse import get_client, observe, propagate_attributes

from agentic_now import db
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from agentic_now.rag.embed import embed_documents

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1500  # characters (~375 tokens)
CHUNK_OVERLAP = 200
MIN_PAGE_CHARS = 50  # skip blank / image-only pages
EMBED_BATCH = 32

BOILERPLATE = re.compile(r"^Access for free at openstax\.org$")
# OpenStax footers: "186     7 • Work Motivation" (even) or "7.2 • Content Theories     187" (odd)
FOOTER = re.compile(r"^(\d+)\s{2,}.*•|•.*\s{2,}(\d+)$")


@dataclass
class Page:
    number: int
    printed_number: int | None
    text: str


@dataclass
class Chunk:
    page: int
    printed_page: int | None
    content: str


@observe(name="parse_pdf", capture_input=False, capture_output=False)
def parse_pdf(path: Path) -> tuple[str, list[Page]]:
    """Extract clean text per page, dropping boilerplate and footers."""
    doc = pymupdf.open(path)
    title = doc.metadata.get("title") or path.stem.replace("-", " ").title()
    pages = []
    for number, pdf_page in enumerate(doc, start=1):
        lines, printed = [], None
        for line in pdf_page.get_text().splitlines():
            line = line.strip()
            if not line or BOILERPLATE.match(line):
                continue
            if footer := FOOTER.search(line):
                printed = int(footer.group(1) or footer.group(2))
                continue
            lines.append(line)
        text = _join_lines(lines)
        if len(text) >= MIN_PAGE_CHARS:
            pages.append(Page(number, printed, text))
    _fill_printed_numbers(pages)
    get_client().update_current_span(
        input={"file": path.name}, output={"title": title, "pages": len(pages)}
    )
    return title, pages


def _fill_printed_numbers(pages: list[Page]) -> None:
    """Pages without a footer (chapter openers) get a number if the PDF-to-printed offset is constant."""
    offsets = {p.number - p.printed_number for p in pages if p.printed_number}
    if len(offsets) == 1:
        offset = offsets.pop()
        for page in pages:
            if page.printed_number is None and page.number > offset:
                page.printed_number = page.number - offset


def _join_lines(lines: list[str]) -> str:
    # PDF text has hard line breaks; join into flowing text. Keep hyphens at line ends ("self-actualization").
    text = ""
    for line in lines:
        text += line if text.endswith("-") or not text else " " + line
    return re.sub(r"\s+", " ", text)


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Sliding window that prefers to cut at sentence ends, then at spaces."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
        space = text.find(" ", start, end)
        start = space + 1 if space != -1 else start
    return chunks


@observe(name="ingest_file", capture_input=False, capture_output=False)
def ingest_file(conn: psycopg.Connection, path: Path) -> None:
    started = time.perf_counter()
    title, pages = parse_pdf(path)
    chunks = [
        Chunk(page.number, page.printed_number, content)
        for page in pages
        for content in chunk_text(page.text)
    ]
    logger.info("%s: parsed %d pages -> %d chunks", title, len(pages), len(chunks))

    vectors = []
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors += embed_documents([c.content for c in chunks[i : i + EMBED_BATCH]])
        if len(vectors) % (EMBED_BATCH * 10) == 0 or len(vectors) == len(chunks):
            logger.info("%s: embedded %d/%d chunks", title, len(vectors), len(chunks))

    # Replace any previous version of this file in one transaction (re-ingest is idempotent).
    with conn.transaction():
        conn.execute("DELETE FROM documents WHERE file_name = %s", (path.name,))
        doc_id = conn.execute(
            "INSERT INTO documents (file_name, title, page_count) VALUES (%s, %s, %s) RETURNING id",
            (path.name, title, len(pymupdf.open(path))),
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, printed_page, chunk_index, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (doc_id, c.page, c.printed_page, i, c.content, v)
                    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
    elapsed = time.perf_counter() - started
    logger.info("%s: stored %d chunks in %.0fs", title, len(chunks), elapsed)
    get_client().update_current_span(
        input={"file": path.name},
        output={"title": title, "pages": len(pages), "chunks": len(chunks)},
    )


def main() -> None:
    setup_logging()
    init_observability()
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/books")
    files = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    if not files:
        sys.exit(f"No PDFs found in {target}")
    db.init_db()
    with db.connect() as conn:
        for path in files:
            with propagate_attributes(
                trace_name=f"ingest {path.name}", tags=["ingest"]
            ):
                ingest_file(conn, path)
    flush()


if __name__ == "__main__":
    main()
