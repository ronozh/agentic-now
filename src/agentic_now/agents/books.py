"""Agentic RAG (iteration 04). The agent decides whether, what and how often to search.

Iteration 07: the final answer is an output function that resolves cited result numbers to
sources itself, so the agent returns a complete `RagAnswer` (also over A2A).

Run: uv run python -m agentic_now.agents.books "question"
"""

import asyncio
import logging
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field

import psycopg
from langfuse import get_client, observe, propagate_attributes
from pydantic_ai import (
    Agent,
    ModelRetry,
    RunContext,
    TextOutput,
    ToolOutput,
    UsageLimits,
)
from pydantic_ai.messages import ToolCallPart

from agentic_now import db
from agentic_now.config import settings
from agentic_now.llm import chat_model
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from agentic_now.rag.answer import RagAnswer
from agentic_now.rag.retrieve import Hit, retrieve

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
You answer questions about management and organizational behavior from the indexed books.
Follow these steps (small models need them spelled out, see .plan/gotcha.md):
1. Call `search` with a precise query. For comparisons or multi-part questions, search once
   per part. Use `read_page` for more context around a result.
2. Call `final_answer` using ONLY the search results. Cite results inline by number, e.g.
   "... five levels [2][4]", and list those numbers in sources.
3. If the results don't answer the question, call `final_answer` with answer "I don't know".
Never answer from your own knowledge. Only for greetings may you skip searching.
"""

MAX_REQUESTS = 8  # loop guard: stops runaway tool calling


@dataclass
class BooksDeps:
    conn: psycopg.Connection
    k: int = 5
    # Every result shown to the LLM, numbered 1..n across all tool calls.
    seen: list[Hit] = field(default_factory=list)
    # Sync tools may run in parallel threads; numbering must not interleave.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set once the agent has been told to search before answering (see require_search).
    nudged: bool = False


def _format(ctx: RunContext[BooksDeps], hits: list[Hit]) -> str:
    """Number results globally across all tool calls, so the LLM can cite any of them."""
    if not hits:
        return "No results."
    blocks = []
    with ctx.deps.lock:
        for hit in hits:
            ctx.deps.seen.append(hit)
            blocks.append(f"[{len(ctx.deps.seen)}] ({hit.source.label})\n{hit.content}")
    return "\n\n".join(blocks)


def final_answer(
    ctx: RunContext[BooksDeps], answer: str, sources: list[int]
) -> RagAnswer:
    """Give the final answer.

    Args:
        answer: Answer based only on the search results, or "I don't know".
        sources: Numbers of the search results that support the answer, e.g. [1, 3].
    """
    seen = ctx.deps.seen
    # Small models answer well-known topics from memory. Send them back to search once;
    # a second answer without searching (e.g. to a greeting) is accepted.
    if not seen and not ctx.deps.nudged:
        ctx.deps.nudged = True
        raise ModelRetry(
            "Search the books first and answer only from the results. "
            "If this was a greeting, just answer it again."
        )
    invalid = [n for n in sources if not 1 <= n <= len(seen)]
    if invalid:
        logger.warning("LLM cited non-existent results %s; ignored", invalid)
    return RagAnswer(
        answer=answer,
        # dict.fromkeys de-duplicates while keeping order (several chunks per page).
        citations=list(
            dict.fromkeys(seen[n - 1].source for n in sources if n not in invalid)
        ),
        retrieved=[hit.source for hit in seen],
    )


def text_answer(ctx: RunContext[BooksDeps], text: str) -> RagAnswer:
    """Small models often reply in plain text instead of calling `final_answer` and ignore
    the retry asking for a tool call. Accept the text; sources come from its [n] markers."""
    sources = list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", text)))
    return final_answer(ctx, text, sources)


agent = Agent(
    chat_model(),
    deps_type=BooksDeps,
    output_type=[ToolOutput(final_answer), TextOutput(text_answer)],
    instructions=INSTRUCTIONS,
    name="books",
)


@agent.tool
def list_books(ctx: RunContext[BooksDeps]) -> list[str]:
    """List the titles of the indexed books."""
    rows = ctx.deps.conn.execute("SELECT title FROM documents ORDER BY title")
    return [row[0] for row in rows]


@agent.tool
def search(ctx: RunContext[BooksDeps], query: str, book: str | None = None) -> str:
    """Semantic search over the books. Returns numbered passages with their source.

    Args:
        query: A precise search query about one topic.
        book: Optional book title (or part of it) to restrict the search.
    """
    hits = retrieve(ctx.deps.conn, query, ctx.deps.k, book)
    if (
        not hits and book
    ):  # the 4B model invents titles; an empty filter reads as "not covered"
        logger.info("no book matches %r; searching all books", book)
        hits = retrieve(ctx.deps.conn, query, ctx.deps.k)
    return _format(ctx, hits)


@agent.tool
def read_page(ctx: RunContext[BooksDeps], book: str, page: int) -> str:
    """Read the full text of one printed page, e.g. to get more context around a search result.

    Args:
        book: Book title (or part of it).
        page: Printed page number, as shown in search results ("p. 186").
    """
    rows = ctx.deps.conn.execute(
        """
        SELECT d.title, c.page, c.printed_page, string_agg(c.content, ' ' ORDER BY c.chunk_index)
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.title ILIKE '%%' || %s || '%%' AND c.printed_page = %s
        GROUP BY d.title, c.page, c.printed_page
        """,
        (book, page),
    ).fetchall()
    return _format(ctx, [Hit(*row, distance=0.0) for row in rows])


@observe(name="books_agent")
async def ask_async(question: str, k: int = 5) -> RagAnswer:
    """Async entry point: callers inside another agent's tool can't use `run_sync`."""
    get_client().update_current_span(metadata={"model": settings.chat_model, "k": k})
    started = time.perf_counter()
    # autocommit: no transaction left open while the LLM thinks for minutes.
    # In a thread: connect retries sleep and must not block the event loop.
    conn = await asyncio.to_thread(db.connect, autocommit=True)
    with conn:
        result = await agent.run(
            question,
            deps=BooksDeps(conn=conn, k=k),
            usage_limits=UsageLimits(request_limit=MAX_REQUESTS),
        )
    tool_calls = [
        f"{part.tool_name}({part.args})"
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    logger.info(
        "agent answered in %.1fs (requests=%d, tools=%s)",
        time.perf_counter() - started,
        result.usage.requests,
        tool_calls,
    )
    return result.output


def ask(question: str, k: int = 5) -> RagAnswer:
    return asyncio.run(ask_async(question, k))


def main() -> None:
    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=str(uuid.uuid4()), user_id="dev", tags=["books-agent", "cli"]
    ):
        result = ask(sys.argv[1])
    flush()
    print(result.answer)
    for citation in result.citations:
        print(f"  - {citation.label}")


if __name__ == "__main__":
    main()
