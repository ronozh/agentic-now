# agentic-now

A hands-on agentic system built with a free, local, open-source stack: RAG over books, an HR database with MCP tools, a semantic layer, multiple agents, observability and evaluation.

## Stack

Python 3.12 · uv · Ollama (local LLMs) · PostgreSQL + pgvector · Pydantic AI · FastMCP · Langfuse · Pydantic Evals

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker, [Ollama](https://ollama.com) (run natively for GPU acceleration on Mac).

```bash
scripts/ollama_serve.sh            # Ollama with 8K context (default 4K silently truncates)
ollama pull qwen3:4b-instruct      # dev default: fast, fits 16 GB; qwen3:8b for quality runs
ollama pull nomic-embed-text

cp .env.example .env
docker compose up -d               # Postgres + pgvector on :5433
uv sync
uv run python -m agentic_now.db    # creates extension + tables
uv run python scripts/hello_agent.py
```

## RAG over books

```bash
scripts/download_books.sh                          # 2 OpenStax PDFs (CC BY-NC-SA) -> data/books/
uv run python -m agentic_now.rag.ingest data/books/ # parse -> chunk -> embed -> pgvector (~3 min)
uv run python -m agentic_now.rag.retrieve "What did Henri Fayol contribute to management theory?"
uv run python -m agentic_now.rag.answer   "What did Henri Fayol contribute to management theory?"
```

Answers cite the printed page and the PDF page, e.g. `Principles of Management, p. 64 (PDF p. 78)`.

## Agentic RAG

```bash
uv run python -m agentic_now.agents.books "Compare Herzberg's two-factor theory with Taylor's scientific management"
```

Instead of one fixed retrieval, the agent decides what to do with its tools (`search`, `read_page`,
`list_books`): it can search several times and read whole pages, and must cite what it used. It refuses questions the books don't cover.

## Observability (Langfuse)

```bash
(cd infra/langfuse && docker compose up -d)   # ~2 GB RAM; UI http://localhost:3000
                                              # login admin@agentic-now.local / agentic-now-local
uv run python -m agentic_now.rag.answer "What is Maslow's hierarchy of needs?"
uv run python scripts/show_trace.py           # latest trace as a tree in the terminal
```

Pydantic AI agent runs, model calls and tool calls are traced automatically; our own steps use `@observe`.
Without Langfuse keys in `.env`, or when Langfuse is stopped, tracing is off and everything still runs.
On 16 GB RAM, keep Langfuse stopped unless you are looking at traces (`docker compose stop`).

## Evaluation

```bash
uv run python -m evals.rag_eval --target retrieval     # hit@k, MRR only: ~2s, deterministic
uv run python -m evals.rag_eval                        # + answer checks + LLM judge: ~6 min
uv run python -m evals.rag_eval --k 10 --label k10     # experiment; compare rows in evals/RESULTS.md
uv run python -m evals.rag_eval --target agent --limit 3   # agentic RAG, smoke test on 3 cases
```

Golden set: `evals/datasets/rag_golden.yaml` (pages verified by hand). Scores are also attached to each case's Langfuse trace.

## Configuration

All settings come from `.env` (see `.env.example`). Swap the chat model without code changes:

```bash
CHAT_MODEL=mistral uv run python scripts/hello_agent.py
```

Logs go to stderr (`LOG_LEVEL=DEBUG|INFO|WARNING`); CLI results go to stdout.

## Roadmap

- [x] 00 Foundation — infra + hello-world agent with a tool
- [x] 01 Basic RAG — PDFs → pgvector → cited answers
- [x] 02 Observability — Langfuse tracing
- [x] 03 Evaluation — golden set + baseline
- [x] 04 Agentic RAG — agent-driven search
- [ ] 05 HR database + MCP — text-to-SQL
- [ ] 06 Semantic layer — governed metrics (headcount, FTE)
- [ ] 07 Multi-agent + A2A
- [ ] 08 API + UI
