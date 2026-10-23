# agentic-now

Learning project: local agentic system (RAG, MCP, semantic layer, multi-agent, observability, evals). Breadth first, then depth.

## Run
- `scripts/ollama_serve.sh` (Ollama natively, 16K ctx) · `docker compose up -d` (pgvector :5433)
- `uv sync` · `uv run ruff check . && uv run ruff format .` · `uv run pytest`
- Evals: `--limit 3` for smoke; full set at tag time. Langfuse (`infra/langfuse`) only when tracing; tracing auto-off if down.

## Conventions
- Settings only via `config.py` (`.env`); models only via `llm.chat_model()`.
- Logging: `logger = logging.getLogger(__name__)`; `setup_logging()` in CLI `main()`. No `print` except CLI results (stdout).
- Agents: Pydantic AI; structured output with `NativeOutput`; thinking off (`openai_reasoning_effort="none"`).
- Dev model `qwen3:4b-instruct` (speed over accuracy); `qwen3:8b` for quality runs; mistral is unreliable at tools.

## Workflow
- Plans: `.plan/NN-*/plan.md` + `validation.md` (gitignored). Log every issue + fix in `.plan/gotcha.md`.
- Per iteration: validate → commit → annotated tag `iter-NN-<name>` → `git push origin main --follow-tags`. Single `main`; never backport to tags.

## Constraints
- 16 GB RAM: keep only needed containers up; Docker VM 4 GB; one chat model loaded; Ollama ctx 8K.
- Responses to user: very concise.
