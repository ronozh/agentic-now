#!/usr/bin/env bash
# Start the full stack: Postgres, Langfuse, HR MCP (HTTP), Books A2A service, API and UI.
# Ollama must already run (scripts/ollama_serve.sh). Ctrl-C stops the app processes.
set -euo pipefail
cd "$(dirname "$0")/.."

curl -sf localhost:11434/api/version >/dev/null || { echo "Ollama is not running: scripts/ollama_serve.sh"; exit 1; }
docker compose up -d
(cd infra/langfuse && docker compose up -d)
# Services check Langfuse once at startup and disable tracing if it isn't up yet.
echo "waiting for Langfuse..."
for _ in $(seq 1 60); do curl -sf localhost:3000/api/public/health >/dev/null && break; sleep 2; done

export BOOKS_AGENT_MODE=a2a HR_MCP_TRANSPORT=http
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT INT TERM
uv run python -m agentic_now.mcp_servers.hr_server --transport http --port 8002 & pids+=($!)
uv run uvicorn agentic_now.a2a.books_service:app --port 8001 --log-level warning & pids+=($!)
uv run uvicorn agentic_now.api.app:app --port 8000 --log-level warning & pids+=($!)
uv run streamlit run ui/streamlit_app.py --server.port 8501 --server.headless true & pids+=($!)

echo "UI http://localhost:8501 · API http://localhost:8000/docs · Langfuse http://localhost:3000"
wait
