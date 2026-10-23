"""Iteration 07: call the Books agent over A2A (send message, poll the task, read the artifact)."""

import asyncio
import logging
import time
import uuid

import httpx
from fasta2a.client import A2AClient
from fasta2a.schema import Message, Part

from agentic_now.config import settings
from agentic_now.rag.answer import RagAnswer

logger = logging.getLogger(__name__)

FINAL_STATES = {"completed", "failed", "canceled", "rejected"}
TIMEOUT_SECONDS = 600  # local LLM on a laptop can be slow
POLL_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30


class A2AError(RuntimeError):
    pass


async def ask_books_remote(question: str, url: str | None = None) -> RagAnswer:
    base_url = url or settings.books_a2a_url
    # httpx's 5 s default is too short when the service shares the laptop with the LLM.
    http_client = httpx.AsyncClient(base_url=base_url, timeout=HTTP_TIMEOUT_SECONDS)
    async with http_client:
        return await _ask(A2AClient(base_url, http_client=http_client), question)


async def _ask(client: A2AClient, question: str) -> RagAnswer:
    message = Message(
        role="user", parts=[Part(text=question)], message_id=str(uuid.uuid4())
    )
    response = await client.send_message(message)
    if "error" in response:
        raise A2AError(f"A2A error: {response['error']}")
    # A2A 1.x wraps the result: {"task": {...}} (or {"message": {...}} for direct replies).
    task = response["result"]["task"]
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while task["status"]["state"] not in FINAL_STATES:
        if time.monotonic() > deadline:
            raise A2AError(f"A2A task {task['id']} timed out after {TIMEOUT_SECONDS}s")
        await asyncio.sleep(POLL_SECONDS)
        task = (await client.get_task(task["id"]))["result"]
    logger.info("A2A task %s finished: %s", task["id"], task["status"]["state"])
    if task["status"]["state"] != "completed":
        raise A2AError(f"A2A task {task['id']} ended as {task['status']['state']}")
    part = task["artifacts"][-1]["parts"][0]
    if "data" in part:  # structured output: {"result": RagAnswer as JSON}
        return RagAnswer.model_validate(part["data"]["result"])
    return RagAnswer(answer=part.get("text", ""), citations=[], retrieved=[])
