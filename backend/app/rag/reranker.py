from __future__ import annotations
from flashrank import Ranker, RerankRequest

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/models/flashrank")
    return _ranker


def rerank(query: str, passages: list[str], top_k: int = 5) -> list[str]:
    if not passages:
        return []
    ranker = _get_ranker()
    passage_dicts = [{"id": i, "text": p} for i, p in enumerate(passages)]
    request = RerankRequest(query=query, passages=passage_dicts)
    results = ranker.rerank(request)
    return [r["text"] for r in results[:top_k]]
