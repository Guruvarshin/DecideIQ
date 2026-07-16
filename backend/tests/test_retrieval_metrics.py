"""
Task 1: Diagnose context_recall gap.

For each golden document:
  1. Sample chunks from the corpus
  2. Generate 3 synthetic queries per chunk via GPT-4o-mini
  3. Retrieve at k=8 (post-rerank) and k=20 (pre-rerank) for each query
  4. LLM-as-judge labels each retrieved chunk as relevant/not
  5. Compute Precision@k, Recall@k, MRR with ranx

If Recall@20 >> Recall@8  -> reranker is demoting correct chunks
If both low               -> chunking/indexing is the bottleneck

Run: docker exec decideiq-backend-1 python tests/test_retrieval_metrics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio, json, random
from openai import AsyncOpenAI
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection, query_dense
from app.rag.bm25_store import build_bm25, bm25_search
from app.rag.reranker import rerank

FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp"),
    ("/app/data/job_offers/offer_b_finedge.txt",  "FinEdge"),
]

SESSION_ID   = "retrieval_metrics_eval"
SAMPLE_CHUNKS = 20      # parent chunks to sample per document
QUERIES_PER_CHUNK = 2   # synthetic queries per chunk
RRF_K        = 60

_oai = AsyncOpenAI()


# ── Synthetic query generation ────────────────────────────────────────────────

async def generate_queries(chunk_text: str) -> list[str]:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[{
            "role": "user",
            "content": (
                f"Given this document excerpt, write {QUERIES_PER_CHUNK} short, specific questions "
                "whose answers can be found directly in the text. "
                "Output one question per line, no numbering.\n\n"
                f"Excerpt:\n{chunk_text[:800]}"
            ),
        }],
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


# ── LLM-as-judge relevance labelling ─────────────────────────────────────────

async def is_relevant(query: str, chunk: str) -> bool:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Chunk: {chunk[:600]}\n\n"
                "Does this chunk contain information that directly helps answer the query? "
                "Answer with a single word: yes or no."
            ),
        }],
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")


# ── RRF retrieval (mirrors retriever.py) ─────────────────────────────────────

def _rrf(lists: list[list[int]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


async def retrieve_at_k(
    session_id: str,
    doc_idx: int,
    chunks,
    query: str,
    k: int,
    rerank_first: bool = False,
) -> list[int]:
    """Returns ordered list of parent_index values (chunk IDs) for top-k results."""
    child_texts = [c.text for c in chunks]
    n = min(40, len(chunks))

    query_emb = await embed_query(query)
    dense_res = query_dense(session_id, doc_idx, query_emb, n_results=n)
    dense_child_ids = [int(id_.split("_")[1]) for id_ in dense_res["ids"][0]]

    bm25 = build_bm25(child_texts)
    sparse_child_ids = [idx for idx, _ in bm25_search(bm25, query, n=n)]

    rrf_scores = _rrf([dense_child_ids, sparse_child_ids])
    ranked_children = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Expand to unique parents
    seen: set[int] = set()
    parent_ids: list[int] = []
    parent_texts: list[str] = []
    for child_id, _ in ranked_children:
        if child_id >= len(chunks):
            continue
        p_idx = chunks[child_id].parent_index
        if p_idx not in seen:
            seen.add(p_idx)
            parent_ids.append(p_idx)
            parent_texts.append(chunks[child_id].parent_text)

    if rerank_first and parent_texts:
        reranked_texts = rerank(query, parent_texts, top_k=k)
        # Map back to parent_ids in reranked order
        text_to_id = {t: i for i, t in zip(parent_ids, parent_texts)}
        return [text_to_id[t] for t in reranked_texts if t in text_to_id]

    return parent_ids[:k]


# ── Metrics (no external deps) ────────────────────────────────────────────────

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k if k else 0.0

def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)

def mrr(retrieved: list[int], relevant: set[int]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

async def evaluate_doc(path: str, label: str, doc_idx: int):
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")

    with open(path, encoding="utf-8") as f:
        raw_text = f.read()

    chunks = build_parent_child_chunks(raw_text)
    embeddings = await embed_texts([c.text for c in chunks])

    delete_doc_collection(SESSION_ID, doc_idx)
    add_chunks(SESSION_ID, doc_idx, chunks, embeddings)

    # Sample unique parent chunks
    unique_parents = list({c.parent_index: c for c in chunks}.values())
    sampled = random.sample(unique_parents, min(SAMPLE_CHUNKS, len(unique_parents)))

    print(f"  Generating synthetic queries for {len(sampled)} sampled chunks...")
    query_tasks = [generate_queries(c.parent_text) for c in sampled]
    query_results = await asyncio.gather(*query_tasks)

    eval_set: list[tuple[str, int]] = []  # (query, source_parent_idx)
    for chunk, queries in zip(sampled, query_results):
        for q in queries:
            eval_set.append((q, chunk.parent_index))

    print(f"  {len(eval_set)} (query, source_chunk) pairs generated.")
    print(f"  Running retrieval at k=8 (post-rerank) and k=20 (pre-rerank)...")

    # Retrieve at multiple k values concurrently
    retrieve_tasks_8  = [retrieve_at_k(SESSION_ID, doc_idx, chunks, q, k=8,  rerank_first=True)  for q, _ in eval_set]
    retrieve_tasks_20 = [retrieve_at_k(SESSION_ID, doc_idx, chunks, q, k=20, rerank_first=False) for q, _ in eval_set]

    retrieved_8,  retrieved_20 = await asyncio.gather(
        asyncio.gather(*retrieve_tasks_8),
        asyncio.gather(*retrieve_tasks_20),
    )

    print(f"  Running LLM-as-judge relevance labelling...")
    # For each (query, retrieved_chunk_id), judge relevance
    # We only judge chunks in the union of k=20 results to save API calls
    judge_cache: dict[tuple[str, int], bool] = {}
    judge_tasks = []
    judge_keys  = []

    for (query, src_pid), ret_20 in zip(eval_set, retrieved_20):
        for pid in ret_20:
            key = (query, pid)
            if key not in judge_cache:
                judge_cache[key] = None
                chunk_text = next((c.parent_text for c in chunks if c.parent_index == pid), "")
                judge_tasks.append(is_relevant(query, chunk_text))
                judge_keys.append(key)

    judge_results = await asyncio.gather(*judge_tasks)
    for key, result in zip(judge_keys, judge_results):
        judge_cache[key] = result

    # Also always mark source chunk as relevant (it generated the query)
    for query, src_pid in eval_set:
        judge_cache[(query, src_pid)] = True

    print(f"  Computing metrics...")
    p8_list, r8_list, r20_list, mrr8_list, mrr20_list = [], [], [], [], []

    for (query, src_pid), ret_8, ret_20 in zip(eval_set, retrieved_8, retrieved_20):
        relevant = {pid for pid in set(ret_20) | {src_pid} if judge_cache.get((query, pid), False)}

        p8_list.append(precision_at_k(ret_8, relevant, 8))
        r8_list.append(recall_at_k(ret_8, relevant, 8))
        r20_list.append(recall_at_k(ret_20, relevant, 20))
        mrr8_list.append(mrr(ret_8, relevant))
        mrr20_list.append(mrr(ret_20, relevant))

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    p8   = avg(p8_list)
    r8   = avg(r8_list)
    r20  = avg(r20_list)
    mrr8 = avg(mrr8_list)

    print(f"\n  Results ({len(eval_set)} queries):")
    print(f"  {'Precision@8':<22}: {p8:.4f}")
    print(f"  {'Recall@8 (post-rerank)':<22}: {r8:.4f}")
    print(f"  {'Recall@20 (pre-rerank)':<22}: {r20:.4f}")
    print(f"  {'MRR@8':<22}: {mrr8:.4f}")

    gap = r20 - r8
    print(f"\n  Recall gap (k20 - k8): {gap:+.4f}")
    if gap > 0.10:
        print("  >> RERANKER is demoting relevant chunks.")
        print("     Recommendation: increase candidates passed to reranker (try k=12 or k=15).")
    elif r20 < 0.70:
        print("  >> RETRIEVAL/CHUNKING is the bottleneck.")
        print("     Recommendation: try 600-char children or retrieve top-40 before RRF.")
    else:
        print("  >> Retrieval and reranking both look healthy.")

    delete_doc_collection(SESSION_ID, doc_idx)
    return {"label": label, "p8": p8, "r8": r8, "r20": r20, "mrr8": mrr8, "gap": gap}


async def main():
    random.seed(42)
    print("=" * 58)
    print("  DecideIQ -- Retrieval Diagnostics (Task 1)")
    print("=" * 58)

    results = []
    for idx, (path, label) in enumerate(FILES):
        r = await evaluate_doc(path, label, idx)
        results.append(r)

    print(f"\n\n{'='*58}")
    print("  SUMMARY")
    print(f"{'='*58}")
    print(f"  {'Doc':<22} {'P@8':>6} {'R@8':>6} {'R@20':>6} {'MRR@8':>6} {'Gap':>6}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['label']:<22} {r['p8']:>6.3f} {r['r8']:>6.3f} {r['r20']:>6.3f} {r['mrr8']:>6.3f} {r['gap']:>+6.3f}")


asyncio.run(main())
