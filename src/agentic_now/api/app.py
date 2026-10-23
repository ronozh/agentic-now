"""Iteration 08: the whole system behind an HTTP API.

Run: uv run uvicorn agentic_now.api.app:app --port 8000      (docs: http://localhost:8000/docs)
"""

import logging
from collections import OrderedDict
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from langfuse import get_client, propagate_attributes
from pydantic import BaseModel, Field

from agentic_now import db
from agentic_now.agents import router
from agentic_now.config import settings
from agentic_now.log import setup_logging
from agentic_now.obs import init_observability

logger = logging.getLogger(__name__)

setup_logging()
init_observability()
app = FastAPI(
    title="agentic-now", description="Books (RAG) + HR (semantic layer) assistant"
)

# Conversation state in memory (lost on restart; Postgres-backed in Phase 2): per session a
# list of turns, each a complete list of messages, so trimming never splits a tool call/result.
SESSIONS: OrderedDict[str, list[list]] = OrderedDict()
MAX_TURNS = 4  # history sent to the LLM; keeps prompts inside the 8K context
MAX_SESSIONS = 100  # least recently used sessions are dropped


def _history(session_id: str) -> list:
    return [
        message
        for turn in SESSIONS.get(session_id, [])[-MAX_TURNS:]
        for message in turn
    ]


def _remember(session_id: str, turn: list) -> None:
    SESSIONS.setdefault(session_id, []).append(turn)
    SESSIONS[session_id] = SESSIONS[session_id][-MAX_TURNS:]
    SESSIONS.move_to_end(session_id)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(router.RouterAnswer):
    trace_id: str | None


class Feedback(BaseModel):
    trace_id: str
    score: Literal[0, 1]  # thumbs down / up
    comment: str | None = Field(default=None, max_length=1000)


# Async: all requests share the server's event loop, so the model's HTTP client is never
# used from two loops, and SESSIONS is only touched between awaits (no lock needed).
@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    langfuse = get_client()
    with (
        propagate_attributes(
            session_id=request.session_id, user_id="api", tags=["api"]
        ),
        langfuse.start_as_current_observation(
            name="chat_turn", input=request.message
        ) as span,
    ):
        trace_id = langfuse.get_current_trace_id()
        try:
            answer, turn = await router.ask_async(
                request.message, _history(request.session_id)
            )
        except Exception as e:
            logger.exception("chat failed")
            span.update(level="ERROR", status_message=str(e))
            raise HTTPException(
                503, f"The assistant is unavailable: {type(e).__name__}"
            ) from e
        span.update(output=answer.answer)
    _remember(request.session_id, turn)
    return ChatResponse(**answer.model_dump(), trace_id=trace_id)


@app.post("/feedback")
def feedback(item: Feedback) -> dict[str, str]:
    get_client().create_score(
        trace_id=item.trace_id,
        name="user_feedback",
        value=item.score,
        comment=item.comment,
    )
    return {"status": "recorded"}


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    SESSIONS.pop(session_id, None)
    return {"status": "reset"}


@app.get("/health")
def health() -> dict[str, str]:
    def check(probe) -> str:
        try:
            probe()
            return "ok"
        except Exception as e:  # noqa: BLE001 - report, never raise: health lists every dependency
            return f"down ({type(e).__name__})"

    def postgres() -> None:
        with db.connect(attempts=1, connect_timeout=3) as conn:
            conn.execute("SELECT 1")

    def ollama() -> None:
        httpx.get(
            settings.ollama_base_url.removesuffix("/v1") + "/api/version", timeout=3
        ).raise_for_status()

    def langfuse() -> None:
        if not get_client().auth_check():
            raise RuntimeError("auth failed")

    return {
        "postgres": check(postgres),
        "ollama": check(ollama),
        "langfuse": check(langfuse),
    }
