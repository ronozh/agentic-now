from functools import cache

import numpy as np
from openai import OpenAI

from agentic_now.config import settings

# nomic-embed-text is trained with task prefixes: documents and queries must be embedded differently.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@cache
def _client() -> OpenAI:
    return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")


def _embed(texts: list[str]) -> list[np.ndarray]:
    response = _client().embeddings.create(model=settings.embed_model, input=texts)
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return _embed([DOC_PREFIX + text for text in texts])


def embed_query(text: str) -> np.ndarray:
    return _embed([QUERY_PREFIX + text])[0]
