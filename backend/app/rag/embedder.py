from __future__ import annotations
from openai import AsyncOpenAI
from langsmith import traceable
from app.core.config import settings

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


@traceable(name="embed_texts", metadata={"model": EMBED_MODEL})
async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = await client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


@traceable(name="embed_query", metadata={"model": EMBED_MODEL})
async def embed_query(text: str) -> list[float]:
    results = await embed_texts([text])
    return results[0]
