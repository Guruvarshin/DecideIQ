from __future__ import annotations
from tavily import AsyncTavilyClient
from app.core.config import settings
from app.rag.grounding import grounding_score, is_grounded

_client: AsyncTavilyClient | None = None

_NOT_MENTIONED = "Not mentioned in available sources."


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    return _client


async def web_search(query: str, max_results: int = 3) -> list[str]:
    """Raw web search — returns content strings or empty list."""
    response = await _get_client().search(query=query, max_results=max_results)
    return [r["content"] for r in response.get("results", []) if r.get("content")]


async def web_search_verified(query: str, max_results: int = 3) -> tuple[list[str], str]:
    """
    Web search with grounding verification.
    Returns (contexts, source) where source is "web" or "not_mentioned".
    If results don't ground to the query (score < threshold), returns the
    sentinel string so the answerer never hallucinates from irrelevant web pages.
    """
    results = await web_search(query, max_results=max_results)

    if not results:
        return [_NOT_MENTIONED], "not_mentioned"

    score = await grounding_score(query, results)
    if not is_grounded(score):
        return [_NOT_MENTIONED], "not_mentioned"

    return results, "web"
