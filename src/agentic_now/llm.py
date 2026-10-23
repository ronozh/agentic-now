from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.ollama import OllamaProvider

from agentic_now.config import settings


def chat_model(
    name: str | None = None, thinking: bool | None = None
) -> OpenAIChatModel:
    """Local chat model served by Ollama. Swap models via CHAT_MODEL, not code."""
    thinking = settings.chat_thinking if thinking is None else thinking
    # Pydantic AI's unified `thinking=False` is dropped for Ollama models (no thinking profile),
    # so send Ollama's OpenAI-compatible switch directly. Non-thinking models ignore it.
    model_settings = (
        OpenAIChatModelSettings()
        if thinking
        else OpenAIChatModelSettings(openai_reasoning_effort="none")
    )
    return OpenAIChatModel(
        name or settings.chat_model,
        provider=OllamaProvider(base_url=settings.ollama_base_url),
        settings=model_settings,
    )
