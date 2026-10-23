from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env to os.environ too: libraries (Pydantic AI, Langfuse) read their own env vars.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5433/agentic"
    # The HR agent's tools connect as a read-only role (sql/hr_roles.sql).
    hr_database_url: str = "postgresql://hr_readonly:hr_readonly@localhost:5433/agentic"
    ollama_base_url: str = "http://localhost:11434/v1"
    chat_model: str = "qwen3:4b-instruct"  # dev speed; qwen3:8b for quality runs
    # Reasoning models (qwen3) are much slower with thinking on.
    chat_thinking: bool = False
    embed_model: str = "nomic-embed-text"
    log_level: str = "INFO"

    # Iteration 07: where the specialists run. local/stdio = in this process / subprocess.
    books_agent_mode: Literal["local", "a2a"] = "local"
    books_a2a_url: str = "http://localhost:8001"
    hr_mcp_transport: Literal["stdio", "http"] = "stdio"
    hr_mcp_url: str = "http://localhost:8002/mcp"

    # Iteration 08: where the UI finds the API.
    api_url: str = "http://localhost:8000"


settings = Settings()
