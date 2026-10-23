"""Iteration 07: router agent. Delegates to the Books agent (in-process or over A2A) and the
HR agent (semantic layer over MCP), then combines their answers.

Run: uv run python -m agentic_now.agents.router "question"
     BOOKS_AGENT_MODE=a2a HR_MCP_TRANSPORT=http uv run python -m agentic_now.agents.router "..."
"""

import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field

from langfuse import get_client, observe, propagate_attributes
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits

from agentic_now.agents import books, hr
from agentic_now.config import settings
from agentic_now.llm import chat_model
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from agentic_now.rag.answer import RagAnswer
from agentic_now.rag.retrieve import Source

logger = logging.getLogger(__name__)

MAX_REQUESTS = 8
OUT_OF_SCOPE = "I can only help with management concepts and our company's HR data."
NOT_FOUND = "The books don't cover this."

INSTRUCTIONS = """\
You are the front door of a company assistant with two specialists:
- ask_books: management and organizational behavior concepts, answered from textbooks.
- ask_hr: numbers about OUR company's workforce (headcount, FTE, leave, part-time, hiring).
Rules:
- Send each part of the question to the right specialist; call both for mixed questions.
- Pass the user's own words (fix typos, resolve "it"/"they" from earlier turns). Do not add
  phrases like "according to textbooks"; they hurt the specialist's search.
- Do not answer from your own knowledge. If neither specialist fits, do not call a tool;
  reply: "I can only help with management concepts and our company's HR data."
- If a tool returns ERROR, tell the user that specialist is unavailable right now.
- Combine the specialists' answers into one answer. Keep book citations and HR definitions.
"""


@dataclass
class RouterDeps:
    agents_used: list[str] = field(default_factory=list)
    citations: list[Source] = field(default_factory=list)
    hr_answers: list[hr.HRAnswer] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class RouterAnswer(BaseModel):
    answer: str
    agents_used: list[str]
    citations: list[Source]
    hr_answers: list[hr.HRAnswer]
    errors: list[str] = []  # specialists that failed (the router still answers)


agent = Agent(
    chat_model(), deps_type=RouterDeps, instructions=INSTRUCTIONS, name="router"
)


async def _ask_books(question: str) -> RagAnswer:
    if settings.books_agent_mode == "a2a":
        from agentic_now.a2a.client import (
            ask_books_remote,  # A2A deps only needed in this mode
        )

        return await ask_books_remote(question)
    return await books.ask_async(question)


# Async tools: a sub-agent can't `run_sync` inside another agent's run (deadlock guard).
@agent.tool
async def ask_books(ctx: RunContext[RouterDeps], question: str) -> str:
    """Ask the Books specialist a question about management or organizational behavior concepts.

    Args:
        question: A complete, self-contained question.
    """
    ctx.deps.agents_used.append("books")
    try:
        result = await _ask_books(question)
        if (
            not result.citations
        ):  # the 4B Books agent sometimes gives up; one retry is cheap
            logger.info("books found nothing; retrying once")
            result = await _ask_books(question)
    except Exception as e:  # a specialist being down must not crash the router
        logger.exception("books specialist failed")
        ctx.deps.errors.append(f"books: {type(e).__name__}")
        return "ERROR: the books specialist is unavailable. Tell the user to try again later."
    if not result.citations:
        return NOT_FOUND
    ctx.deps.citations += [c for c in result.citations if c not in ctx.deps.citations]
    sources = "; ".join(c.label for c in result.citations) or "none"
    return f"{result.answer}\nSources: {sources}"


@agent.tool
async def ask_hr(ctx: RunContext[RouterDeps], question: str) -> str:
    """Ask the HR specialist about the company's workforce numbers (headcount, FTE, leave, hiring).

    Args:
        question: A complete, self-contained question.
    """
    ctx.deps.agents_used.append("hr")
    try:
        result = await hr.ask_async(question, mode="semantic")
    except Exception as e:
        logger.exception("hr specialist failed")
        ctx.deps.errors.append(f"hr: {type(e).__name__}")
        return (
            "ERROR: the HR specialist is unavailable. Tell the user to try again later."
        )
    ctx.deps.hr_answers.append(result)
    return f"{result.answer}\nValue: {result.value}"


@observe(name="router")
async def ask_async(
    question: str, message_history: list | None = None
) -> tuple[RouterAnswer, list]:
    """Answer one turn. Returns the answer and this turn's messages (for conversation history)."""
    get_client().update_current_span(
        metadata={
            "model": settings.chat_model,
            "books_agent_mode": settings.books_agent_mode,
            "hr_mcp_transport": settings.hr_mcp_transport,
        }
    )
    started = time.perf_counter()
    deps = RouterDeps()
    result = await agent.run(
        question,
        deps=deps,
        message_history=message_history,
        usage_limits=UsageLimits(request_limit=MAX_REQUESTS),
    )
    logger.info(
        "router answered in %.1fs (requests=%d, agents=%s)",
        time.perf_counter() - started,
        result.usage.requests,
        deps.agents_used,
    )
    output = result.output
    if deps.agents_used and output.startswith(OUT_OF_SCOPE[:20]):
        # The 4B router answers "not found" with the out-of-scope line; it was in scope.
        output = NOT_FOUND
    answer = RouterAnswer(
        answer=output,
        agents_used=sorted(set(deps.agents_used)),
        citations=deps.citations,
        hr_answers=deps.hr_answers,
        errors=deps.errors,
    )
    return answer, result.new_messages()


def ask(
    question: str, message_history: list | None = None
) -> tuple[RouterAnswer, list]:
    return asyncio.run(ask_async(question, message_history))


def main() -> None:
    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=str(uuid.uuid4()), user_id="dev", tags=["router", "cli"]
    ):
        result, _ = ask(sys.argv[1])
    flush()
    print(result.answer)
    print(f"  agents: {result.agents_used}")
    for citation in result.citations:
        print(f"  - {citation.label}")


if __name__ == "__main__":
    main()
