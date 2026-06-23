from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import AsyncTavilyClient
from app.core.config import settings
from app.rag.grounding import grounding_score, is_grounded

_client: AsyncTavilyClient | None = None
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_NOT_MENTIONED = "Not mentioned in available sources."

_RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "Question: {question}\n\n"
    "Web search result:\n{context}\n\n"
    "Does this web result contain specific, direct information that answers the question above? "
    "It must answer THIS specific question — not just be topically related.\n"
    "If the question is about a specific company, person, or document and the result is generic "
    "advice or unrelated content, answer no.\n"
    "Answer with a single word: yes or no."
)

_relevance_chain = _RELEVANCE_PROMPT | _llm


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    return _client


async def web_search(query: str, max_results: int = 3) -> list[str]:
    response = await _get_client().search(query=query, max_results=max_results)
    return [r["content"] for r in response.get("results", []) if r.get("content")]


async def _is_web_result_relevant(query: str, contexts: list[str]) -> bool:
    """
    LLM relevance check on web results. Cosine grounding alone is too weak —
    a query like 'What is the best company?' triggers web search and Tavily returns
    generic 'how to compare companies' articles that pass cosine threshold 0.35
    but don't actually answer the question. This check asks the LLM directly.
    """
    result = await _relevance_chain.ainvoke({
        "question": query,
        "context": "\n\n".join(contexts)[:1200],
    })
    return result.content.strip().lower().startswith("y")


async def web_search_verified(query: str, max_results: int = 3) -> tuple[list[str], str]:
    """
    Web search with two-stage verification:
    1. Cosine grounding check (fast, embedding-based) — catches completely off-topic results
    2. LLM relevance check (slower, meaning-based) — catches topically related but useless results
       e.g. generic advice articles for document-specific questions
    """
    results = await web_search(query, max_results=max_results)

    if not results:
        return [_NOT_MENTIONED], "not_mentioned"

    score = await grounding_score(query, results)
    if not is_grounded(score):
        return [_NOT_MENTIONED], "not_mentioned"

    # Stage 2: LLM relevance check — guards against generic web results
    if not await _is_web_result_relevant(query, results):
        return [_NOT_MENTIONED], "not_mentioned"

    return results, "web"
