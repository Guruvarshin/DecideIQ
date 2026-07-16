"""
Task 1b: Retrieval diagnostics on insurance plan documents.

These are real-world length documents (87-108 parent chunks each vs 2 for job offers),
giving statistically meaningful Precision@k, Recall@k, MRR results.

Run: docker exec decideiq-backend-1 python tests/test_retrieval_insurance.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio, random
from openai import AsyncOpenAI
from app.ingestion.pdf_parser import parse_pdf
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection, query_dense
from app.rag.bm25_store import build_bm25, bm25_search
from app.rag.reranker import rerank

FILES = [
    "/app/data/insurance/hdfc_ergo_optima_restore.pdf",
    "/app/data/insurance/niva_bupa_reassure_3.pdf",
    "/app/data/insurance/star_health_family_optima.pdf",
]

SESSION_ID     = "insurance_retrieval_eval"
SAMPLE_CHUNKS  = 20
QUERIES_PER    = 2
RRF_K          = 60
K_VALUES       = [3, 5, 8, 12, 20]

_oai = AsyncOpenAI()


async def generate_queries(chunk_text: str) -> list[str]:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini", temperature=0.4,
        messages=[{"role": "user", "content": (
            f"Given this insurance policy excerpt, write {QUERIES_PER} short specific questions "
            "whose answers can be found directly in this text. "
            "Output one question per line, no numbering.\n\n"
            f"Excerpt:\n{chunk_text[:800]}"
        )}],
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


async def is_relevant(query: str, chunk: str) -> bool:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": (
            f"Query: {query}\n\nChunk: {chunk[:600]}\n\n"
            "Does this chunk contain information that directly helps answer the query? "
            "Answer with a single word: yes or no."
        )}],
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")


async def run_judge_batched(tasks: list, keys: list, batch_size: int = 20) -> dict:
    """Run judge tasks in batches to stay under RPM 500 limit."""
    results = {}
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_keys  = keys[i:i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        for key, res in zip(batch_keys, batch_results):
            results[key] = res if isinstance(res, bool) else False
        if i + batch_size < len(tasks):
            await asyncio.sleep(3)  # 3s pause between batches keeps RPM well under 500
    return results


def _rrf(lists: list[list[int]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


async def retrieve_candidates(session_id, doc_idx, chunks, query, n=40) -> list[int]:
    """Returns ordered parent_index list (pre-rerank, up to n unique parents)."""
    child_texts = [c.text for c in chunks]
    cap = min(n * 2, len(chunks))

    query_emb = await embed_query(query)
    dense_res = query_dense(session_id, doc_idx, query_emb, n_results=cap)
    dense_child_ids = [int(id_.split("_")[1]) for id_ in dense_res["ids"][0]]

    bm25 = build_bm25(child_texts)
    sparse_child_ids = [idx for idx, _ in bm25_search(bm25, query, n=cap)]

    rrf_scores = _rrf([dense_child_ids, sparse_child_ids])
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    seen: set[int] = set()
    parent_ids, parent_texts = [], []
    for child_id, _ in ranked:
        if child_id >= len(chunks):
            continue
        p_idx = chunks[child_id].parent_index
        if p_idx not in seen:
            seen.add(p_idx)
            parent_ids.append(p_idx)
            parent_texts.append(chunks[child_id].parent_text)

    return parent_ids, parent_texts


def precision_at_k(retrieved, relevant, k):
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k if k else 0.0

def recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)

def mrr_score(retrieved, relevant):
    for i, r in enumerate(retrieved, 1):
        if r in relevant: return 1.0 / i
    return 0.0


async def evaluate_doc(path: str, doc_idx: int):
    name = path.split("/")[-1].replace(".pdf", "")
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    with open(path, "rb") as f:
        result = parse_pdf(f.read(), path.split("/")[-1])
    raw_text = result["raw_text"]

    chunks = build_parent_child_chunks(raw_text)
    unique_parent_map = {}
    for c in chunks:
        if c.parent_index not in unique_parent_map:
            unique_parent_map[c.parent_index] = c
    unique_parents = list(unique_parent_map.values())
    n_parents = len(unique_parents)

    print(f"  {len(chunks)} child chunks, {n_parents} parent chunks")

    embeddings = await embed_texts([c.text for c in chunks])
    delete_doc_collection(SESSION_ID, doc_idx)
    add_chunks(SESSION_ID, doc_idx, chunks, embeddings)

    sampled = random.sample(unique_parents, min(SAMPLE_CHUNKS, n_parents))
    print(f"  Generating queries for {len(sampled)} sampled chunks...")

    query_results = await asyncio.gather(*[generate_queries(c.parent_text) for c in sampled])

    eval_set = []
    for chunk, queries in zip(sampled, query_results):
        for q in queries:
            eval_set.append((q, chunk.parent_index))
    print(f"  {len(eval_set)} queries generated.")

    print(f"  Retrieving candidates (pre-rerank)...")
    cand_tasks = [retrieve_candidates(SESSION_ID, doc_idx, chunks, q, n=40) for q, _ in eval_set]
    cand_results = await asyncio.gather(*cand_tasks)

    print(f"  Running LLM-as-judge...")
    judge_cache: dict[tuple, bool] = {}
    judge_tasks, judge_keys = [], []
    for (query, src_pid), (parent_ids, _) in zip(eval_set, cand_results):
        for pid in parent_ids[:20]:
            key = (query, pid)
            if key not in judge_cache:
                judge_cache[key] = None
                chunk_text = unique_parent_map.get(pid)
                text = chunk_text.parent_text if chunk_text else ""
                judge_tasks.append(is_relevant(query, text))
                judge_keys.append(key)
    judge_map = await run_judge_batched(judge_tasks, judge_keys, batch_size=20)
    judge_cache.update(judge_map)
    # source chunk always relevant
    for query, src_pid in eval_set:
        judge_cache[(query, src_pid)] = True

    print(f"  Computing metrics across k={K_VALUES}...")
    # For each query, get relevant set from judge labels
    metrics = {k: {"p": [], "r": [], "mrr": []} for k in K_VALUES}

    for (query, src_pid), (parent_ids, parent_texts) in zip(eval_set, cand_results):
        relevant = {pid for pid in (set(parent_ids[:20]) | {src_pid})
                    if judge_cache.get((query, pid), False)}
        if not relevant:
            continue

        # Pre-rerank: just use RRF order
        for k in K_VALUES:
            metrics[k]["p"].append(precision_at_k(parent_ids, relevant, k))
            metrics[k]["r"].append(recall_at_k(parent_ids, relevant, k))
            metrics[k]["mrr"].append(mrr_score(parent_ids, relevant))

        # Post-rerank at k=8 (what the actual pipeline serves)
        reranked_texts = rerank(query, parent_texts[:20], top_k=8)
        text_to_pid = {t: p for p, t in zip(parent_ids, parent_texts)}
        reranked_pids = [text_to_pid[t] for t in reranked_texts if t in text_to_pid]
        metrics["rerank8"] = metrics.get("rerank8", {"p": [], "r": [], "mrr": []})
        metrics["rerank8"]["p"].append(precision_at_k(reranked_pids, relevant, 8))
        metrics["rerank8"]["r"].append(recall_at_k(reranked_pids, relevant, 8))
        metrics["rerank8"]["mrr"].append(mrr_score(reranked_pids, relevant))

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    print(f"\n  {'k':<12} {'Precision@k':>12} {'Recall@k':>10} {'MRR':>8}")
    print(f"  {'-'*44}")
    for k in K_VALUES:
        p = avg(metrics[k]["p"])
        r = avg(metrics[k]["r"])
        m = avg(metrics[k]["mrr"])
        print(f"  pre-rerank k={k:<3}  {p:>12.4f} {r:>10.4f} {m:>8.4f}")

    if "rerank8" in metrics:
        p = avg(metrics["rerank8"]["p"])
        r = avg(metrics["rerank8"]["r"])
        m = avg(metrics["rerank8"]["mrr"])
        print(f"  post-rerank k=8   {p:>12.4f} {r:>10.4f} {m:>8.4f}  << actual pipeline")

    # Diagnose
    r8_pre  = avg(metrics[8]["r"])
    r8_post = avg(metrics.get("rerank8", {}).get("r", [0]))
    r20_pre = avg(metrics[20]["r"])
    gap = r8_pre - r8_post

    print(f"\n  Recall@8 pre-rerank:  {r8_pre:.4f}")
    print(f"  Recall@8 post-rerank: {r8_post:.4f}  (gap: {gap:+.4f})")
    print(f"  Recall@20 pre-rerank: {r20_pre:.4f}")

    if gap > 0.08:
        print(f"\n  >> RERANKER is demoting relevant chunks (gap={gap:+.3f})")
        print(f"     Fix: increase candidates fed to reranker (try passing top-20 instead of top-8)")
    elif r20_pre > r8_pre + 0.08:
        print(f"\n  >> RETRIEVAL bottleneck: more candidates helps (R@20={r20_pre:.3f} vs R@8={r8_pre:.3f})")
        print(f"     Fix: retrieve top-20 from RRF, then rerank down to 8")
    elif r8_post < 0.65:
        print(f"\n  >> CHUNKING bottleneck: retrieval misses evidence even at k=20")
        print(f"     Fix: try 600-char children or semantic chunking")
    else:
        print(f"\n  >> Pipeline looks healthy for this document type.")

    delete_doc_collection(SESSION_ID, doc_idx)
    return {k: {"p": avg(metrics[k]["p"]), "r": avg(metrics[k]["r"]), "mrr": avg(metrics[k]["mrr"])}
            for k in K_VALUES}


async def main():
    random.seed(42)
    print("=" * 60)
    print("  DecideIQ -- Retrieval Diagnostics on Insurance Plans")
    print("=" * 60)

    all_results = {}
    for idx, path in enumerate(FILES):
        name = path.split("/")[-1].replace(".pdf", "")
        all_results[name] = await evaluate_doc(path, idx)

    print(f"\n\n{'='*60}")
    print("  SUMMARY -- Recall@k across all insurance documents")
    print(f"{'='*60}")
    print(f"  {'Document':<35} " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    print(f"  {'-'*58}")
    for name, res in all_results.items():
        row = f"  {name:<35} " + "  ".join(f"{res[k]['r']:.3f}" for k in K_VALUES)
        print(row)

    print()

asyncio.run(main())
