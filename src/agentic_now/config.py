from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env to os.environ too: libraries (Pydantic AI, Langfuse) read their own env vars.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5433/agentic"
    ollama_base_url: str = "http://localhost:11434/v1"
    chat_model: str = "qwen3:8b"
    # Reasoning models (qwen3) are much slower with thinking on.
    chat_thinking: bool = False
    embed_model: str = "nomic-embed-text"
    log_level: str = "INFO"


settings = Settings()
