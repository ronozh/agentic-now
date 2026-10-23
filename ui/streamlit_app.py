"""Iteration 08: chat UI over the API.

Run: uv run streamlit run ui/streamlit_app.py      (needs the API on :8000)
"""

import uuid

import httpx
import streamlit as st

from agentic_now.config import settings

API = settings.api_url
TIMEOUT = 900  # seconds: a local LLM answering a mixed question can take minutes

st.set_page_config(page_title="agentic-now", page_icon="📚")
st.title("agentic-now")
st.caption(
    "Management books (RAG) + company HR data (semantic layer). Local LLM: answers can take a minute."
)


@st.cache_data(ttl=30)  # every rerun (e.g. a feedback click) would probe all services
def health() -> dict:
    return httpx.get(f"{API}/health", timeout=10).json()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.turns = []  # [{"question", "response"}]

with st.sidebar:
    if st.button("New conversation"):
        try:
            httpx.delete(f"{API}/sessions/{st.session_state.session_id}")
        except httpx.HTTPError:
            pass  # the server-side history is dropped on restart anyway
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.turns = []
        st.rerun()
    try:
        st.json(health(), expanded=False)
    except httpx.HTTPError:
        st.error("API not reachable")


def show(response: dict, index: int) -> None:
    st.markdown(response["answer"])
    if response["agents_used"]:
        st.caption("Agents: " + ", ".join(response["agents_used"]))
    if response["citations"] or response["hr_answers"]:
        with st.expander("Sources"):
            for c in response["citations"]:
                page = (
                    f"p. {c['printed_page']} (PDF p. {c['page']})"
                    if c["printed_page"]
                    else f"PDF p. {c['page']}"
                )
                st.markdown(f"- {c['title']}, {page}")
            for h in response["hr_answers"]:
                st.markdown(f"- HR: {h['answer']}")
    if response.get("trace_id"):
        up, down, _ = st.columns([1, 1, 8])
        for column, score, label in ((up, 1, "👍"), (down, 0, "👎")):
            if column.button(label, key=f"fb-{index}-{score}"):
                try:
                    httpx.post(
                        f"{API}/feedback",
                        json={"trace_id": response["trace_id"], "score": score},
                    ).raise_for_status()
                    st.toast("Thanks for the feedback")
                except httpx.HTTPError as e:
                    st.error(f"Feedback not recorded: {e}")


for i, turn in enumerate(st.session_state.turns):
    st.chat_message("user").markdown(turn["question"])
    with st.chat_message("assistant"):
        show(turn["response"], i)

if question := st.chat_input("Ask about management concepts or our workforce"):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            r = httpx.post(
                f"{API}/chat",
                json={"session_id": st.session_state.session_id, "message": question},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:  # the API explains what is down
            try:
                st.error(e.response.json()["detail"])
            except (ValueError, KeyError, TypeError):
                st.error(
                    f"Request failed ({e.response.status_code}): {e.response.text}"
                )
            st.stop()
        except httpx.HTTPError as e:
            st.error(f"API not reachable: {e}")
            st.stop()
    st.session_state.turns.append({"question": question, "response": r.json()})
    st.rerun()
