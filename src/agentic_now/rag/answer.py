"""Iteration 01 generation: retrieve chunks, ask the LLM to answer from them only, with citations.

Run: uv run python -m agentic_now.rag.answer "question"
"""

import logging
import sys
import time
import uuid

from langfuse import get_client, observe, propagate_attributes
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

from agentic_now import db
from agentic_now.config import settings
from agentic_now.llm import chat_model
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from agentic_now.rag.retrieve import Hit, Source, retrieve

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
You answer questions using ONLY the numbered context blocks provided.
- If the context does not contain the answer, set answer to "I don't know" and sources to [].
- Do not use outside knowledge.
- List in sources the numbers of the blocks that support the answer.
"""


class LLMAnswer(BaseModel):
    """What the LLM returns. It cites block numbers only, so it cannot invent sources."""

    answer: str = Field(
        description='Answer based only on the context, or "I don\'t know".'
    )
    sources: list[int] = Field(
        description="Numbers of the supporting context blocks, e.g. [1, 3]."
    )


class RagAnswer(BaseModel):
    answer: str
    citations: list[Source]
    retrieved: list[Source]  # what the LLM was shown, in rank order (used by evals)


# NativeOutput: Ollama constrains decoding to the JSON schema, so output is always valid JSON.
# The default mode lets the model write JSON freely; qwen3 sometimes answers in prose instead.
agent = Agent(
    chat_model(), output_type=NativeOutput(LLMAnswer), instructions=INSTRUCTIONS
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = "\n\n".join(
        f"[{i}] ({hit.source.label})\n{hit.content}"
        for i, hit in enumerate(hits, start=1)
    )
    return f"Context:\n\n{blocks}\n\nQuestion: {question}"


@observe(name="rag_answer")
def answer(question: str, k: int = 5) -> RagAnswer:
    get_client().update_current_span(metadata={"model": settings.chat_model, "k": k})
    with db.connect() as conn:
        hits = retrieve(conn, question, k)
    started = time.perf_counter()
    result = agent.run_sync(build_prompt(question, hits))
    usage = result.usage
    logger.info(
        "LLM answered in %.1fs (requests=%d, input_tokens=%d, output_tokens=%d)",
        time.perf_counter() - started,
        usage.requests,
        usage.input_tokens,
        usage.output_tokens,
    )
    # Map block numbers back to real sources in code; ignore numbers that don't exist.
    invalid = [n for n in result.output.sources if not 1 <= n <= len(hits)]
    if invalid:
        logger.warning("LLM cited non-existent context blocks %s; ignored", invalid)
    # dict.fromkeys de-duplicates while keeping order (several chunks can come from one page).
    citations = list(
        dict.fromkeys(
            hits[n - 1].source for n in result.output.sources if 1 <= n <= len(hits)
        )
    )
    return RagAnswer(
        answer=result.output.answer,
        citations=citations,
        retrieved=[hit.source for hit in hits],
    )


def main() -> None:
    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=str(uuid.uuid4()), user_id="dev", tags=["rag", "cli"]
    ):
        result = answer(sys.argv[1])
    flush()
    print(result.answer)
    for citation in result.citations:
        print(f"  - {citation.label}")


if __name__ == "__main__":
    main()
