import numpy as np
from langsmith import traceable
from app.rag.embedder import embed_query, embed_texts

GROUNDING_THRESHOLD = 0.35


@traceable(name="grounding_score", metadata={"threshold": GROUNDING_THRESHOLD})
async def grounding_score(query: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    q_emb = np.array(await embed_query(query))
    ctx_embs = np.array(await embed_texts(contexts))
    avg_ctx = ctx_embs.mean(axis=0)
    score = float(np.dot(q_emb, avg_ctx) / (np.linalg.norm(q_emb) * np.linalg.norm(avg_ctx)))
    return round(score, 4)


def is_grounded(score: float) -> bool:
    return score >= GROUNDING_THRESHOLD
