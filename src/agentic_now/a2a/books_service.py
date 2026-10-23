"""Iteration 07: the Books agent as an A2A service.

Run:   uv run uvicorn agentic_now.a2a.books_service:app --port 8001
Card:  curl -s localhost:8001/.well-known/agent-card.json
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fasta2a.broker import InMemoryBroker
from fasta2a.pydantic_ai import agent_to_a2a
from fasta2a.pydantic_ai._bridge import AgentWorker, worker_lifespan
from fasta2a.schema import TaskSendParams
from fasta2a.storage import InMemoryStorage
from opentelemetry import context, propagate
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
        # In a thread: connect retries sleep and would stall the service's only event loop.
        conn = await asyncio.to_thread(db.connect, autocommit=True)
        with conn:
            async with super().iter(
                *args,
                deps=deps or BooksDeps(conn=conn),
                usage_limits=usage_limits or UsageLimits(request_limit=MAX_REQUESTS),
                **kwargs,
            ) as run:
                yield run


class TracedAgentWorker(AgentWorker):
    """Continue the caller's trace from the `traceparent` in the message metadata
    (fasta2a doesn't pass message metadata to the agent)."""

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        history = (task or {}).get("history") or [{}]
        token = context.attach(propagate.extract(history[-1].get("metadata") or {}))
        try:
            await super().run_task(params)
        finally:
            context.detach(token)


setup_logging()
init_observability()

agent = BooksServiceAgent(books_agent)
storage, broker = InMemoryStorage(), InMemoryBroker()
worker = TracedAgentWorker(agent=agent, broker=broker, storage=storage)
app = agent_to_a2a(
    agent,
    storage=storage,
    broker=broker,
    lifespan=partial(worker_lifespan, worker=worker, agent=agent),
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
