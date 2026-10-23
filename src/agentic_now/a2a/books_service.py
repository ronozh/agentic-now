"""Iteration 07: the Books agent as an A2A service.

Run:   uv run uvicorn agentic_now.a2a.books_service:app --port 8001
Card:  curl -s localhost:8001/.well-known/agent-card.json
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fasta2a.pydantic_ai import agent_to_a2a
from pydantic_ai import UsageLimits
from pydantic_ai.agent import WrapperAgent

from agentic_now import db
from agentic_now.agents.books import MAX_REQUESTS, BooksDeps
from agentic_now.agents.books import agent as books_agent
from agentic_now.config import settings
from agentic_now.log import setup_logging
from agentic_now.obs import init_observability


class BooksServiceAgent(WrapperAgent):
    """The A2A worker runs agents without deps; give every run a DB connection and a loop guard."""

    @asynccontextmanager
    async def iter(
        self, *args: Any, deps: Any = None, usage_limits: Any = None, **kwargs: Any
    ) -> AsyncIterator:
        with db.connect(autocommit=True) as conn:
            async with super().iter(
                *args,
                deps=deps or BooksDeps(conn=conn),
                usage_limits=usage_limits or UsageLimits(request_limit=MAX_REQUESTS),
                **kwargs,
            ) as run:
                yield run


setup_logging()
init_observability()

app = agent_to_a2a(
    BooksServiceAgent(books_agent),
    name="books",
    url=settings.books_a2a_url,
    description="Answers management and organizational behavior questions from indexed textbooks, with page citations.",
    skills=[
        {
            "id": "answer_from_books",
            "name": "Answer from books",
            "description": "Searches two OpenStax textbooks and answers with citations (book, page).",
            "tags": ["rag", "management", "organizational-behavior"],
            "examples": ["What are the levels of Maslow's hierarchy of needs?"],
            "input_modes": ["text/plain"],
            "output_modes": ["application/json"],
        }
    ],
)
