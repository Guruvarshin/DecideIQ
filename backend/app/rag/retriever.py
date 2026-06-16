from __future__ import annotations
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_query
from app.rag.vector_store import query_dense
from app.rag.bm25_store import build_bm25, bm25_search

RRF_K = 60
DEFAULT_TOP_K = 5


def _rrf(lists: list[list[int]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


async def retrieve(
    session_id: str,
    doc_idx: int,
    raw_text: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[str]:
    chunks = build_parent_child_chunks(raw_text)
    child_texts = [c.text for c in chunks]
    n = min(20, len(chunks))

    query_emb = query_embedding if query_embedding is not None else await embed_query(query)
    dense_res = query_dense(session_id, doc_idx, query_emb, n_results=n)
    dense_child_ids = [int(id_.split("_")[1]) for id_ in dense_res["ids"][0]]

    bm25 = build_bm25(child_texts)
    sparse_ranked = bm25_search(bm25, query, n=n)
    sparse_child_ids = [idx for idx, _ in sparse_ranked]

    rrf_scores = _rrf([dense_child_ids, sparse_child_ids])
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    seen: set[int] = set()
    parent_texts: list[str] = []
    for child_id, _ in ranked:
        if child_id >= len(chunks):
            continue
        p_idx = chunks[child_id].parent_index
        if p_idx not in seen:
            seen.add(p_idx)
            parent_texts.append(chunks[child_id].parent_text)
        if len(parent_texts) >= top_k:
            break

    return parent_texts
