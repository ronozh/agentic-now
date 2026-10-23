# agentic-now

A hands-on agentic system built with a free, local, open-source stack: RAG over books, an HR database with MCP tools, a semantic layer, multiple agents, observability and evaluation.

## Stack

Python 3.12 · uv · Ollama (local LLMs) · PostgreSQL + pgvector · Pydantic AI · FastMCP · Langfuse · Pydantic Evals

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker, [Ollama](https://ollama.com) (run natively for GPU acceleration on Mac).

```bash
ollama serve                       # or run the Ollama app
ollama pull qwen3:8b
ollama pull nomic-embed-text

cp .env.example .env
docker compose up -d               # Postgres + pgvector on :5433
uv sync
uv run python -m agentic_now.db    # enables pgvector
uv run python scripts/hello_agent.py
```

## Configuration

All settings come from `.env` (see `.env.example`). Swap the chat model without code changes:

```bash
CHAT_MODEL=mistral uv run python scripts/hello_agent.py
```

## Roadmap

- [x] 00 Foundation — infra + hello-world agent with a tool
- [ ] 01 Basic RAG — PDFs → pgvector → cited answers
- [ ] 02 Observability — Langfuse tracing
- [ ] 03 Evaluation — golden set + baseline
- [ ] 04 Agentic RAG — agent-driven search
- [ ] 05 HR database + MCP — text-to-SQL
- [ ] 06 Semantic layer — governed metrics (headcount, FTE)
- [ ] 07 Multi-agent + A2A
- [ ] 08 API + UI
