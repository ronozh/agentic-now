"""API tests without an LLM: the router is replaced by a stub that records the history it got."""

import pytest
from fastapi.testclient import TestClient

from agentic_now.agents import router
from agentic_now.api import app as api


@pytest.fixture
def client(monkeypatch):
    calls = []

    async def fake_ask(message, history):
        calls.append((message, list(history)))
        answer = router.RouterAnswer(
            answer=f"echo: {message}", agents_used=[], citations=[], hr_answers=[]
        )
        return answer, [f"turn {len(calls)}"]  # stand-in for this turn's messages

    monkeypatch.setattr(api.router, "ask_async", fake_ask)
    api.SESSIONS.clear()
    test_client = TestClient(api.app)
    test_client.calls = calls
    return test_client


def test_chat_returns_answer(client):
    response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert response.status_code == 200
    assert response.json()["answer"] == "echo: hi"


def test_history_is_per_session_and_trimmed(client):
    for i in range(api.MAX_TURNS + 2):
        client.post("/chat", json={"session_id": "s1", "message": f"m{i}"})
    client.post("/chat", json={"session_id": "s2", "message": "other"})
    *_, last_s1, s2 = client.calls
    assert len(last_s1[1]) == api.MAX_TURNS  # only the last MAX_TURNS turns are sent
    assert s2[1] == []  # sessions don't leak into each other


def test_reset_clears_history(client):
    client.post("/chat", json={"session_id": "s1", "message": "a"})
    client.delete("/sessions/s1")
    client.post("/chat", json={"session_id": "s1", "message": "b"})
    assert client.calls[-1][1] == []


def test_router_failure_is_503(client, monkeypatch):
    async def broken(message, history):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(api.router, "ask_async", broken)
    response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert response.status_code == 503


def test_invalid_request_is_422(client):
    assert (
        client.post("/chat", json={"session_id": "", "message": "hi"}).status_code
        == 422
    )
