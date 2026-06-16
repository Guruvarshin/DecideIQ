from rank_bm25 import BM25Okapi


def build_bm25(texts: list[str]) -> BM25Okapi:
    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)


def bm25_search(bm25: BM25Okapi, query: str, n: int = 20) -> list[tuple[int, float]]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:n]
