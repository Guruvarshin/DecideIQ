"""
Task 3: Chunking experiment — 400-char vs 600-char children on insurance docs.

Optimizations to stay under daily RPD limit:
  - 10 sampled chunks per doc (was 15), 1 query per chunk (was 2)
  - Judge cache SHARED across both configs — parent chunks are identical
    (only the child splitter changes), so (query, parent_id) relevance is config-independent.
    This halves judge API calls compared to running each config independently.

Run: docker exec decideiq-backend-1 python tests/test_chunking_experiment.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio, random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from app.ingestion.pdf_parser import parse_pdf
from app.rag.chunker import Chunk
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection, query_dense
from app.rag.bm25_store import build_bm25, bm25_search

FILES = [
    "/app/data/insurance/hdfc_ergo_optima_restore.pdf",
    "/app/data/insurance/niva_bupa_reassure_3.pdf",
    "/app/data/insurance/star_health_family_optima.pdf",
]

SESSION_ID    = "chunking_experiment"
SAMPLE_CHUNKS = 10   # reduced from 15
QUERIES_PER   = 1    # reduced from 2 — one query per source chunk is enough
RRF_K         = 60
K_VALUES      = [5, 8, 20]

CONFIGS = [
    {"name": "400-char (baseline)", "child_chars": 400, "child_overlap": 60},
    {"name": "600-char (experiment)", "child_chars": 600, "child_overlap": 80},
]

_oai = AsyncOpenAI()


def build_chunks(text: str, child_chars: int, child_overlap: int) -> list[Chunk]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chars, chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parents = parent_splitter.split_text(text)
    chunks = []
    for p_idx, parent_text in enumerate(parents):
        for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):
            chunks.append(Chunk(
                text=child_text, parent_text=parent_text,
                child_index=c_idx, parent_index=p_idx,
            ))
    return chunks


async def generate_queries(chunk_text: str) -> list[str]:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini", temperature=0.3,
        messages=[{"role": "user", "content": (
            f"Write {QUERIES_PER} short specific question(s) whose answer is directly in this text. "
            "Output one question per line, no numbering.\n\n"
            f"Text:\n{chunk_text[:700]}"
        )}],
    )
    lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
    return lines[:QUERIES_PER]


async def is_relevant(query: str, chunk: str) -> bool:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": (
            f"Query: {query}\nChunk: {chunk[:500]}\n"
            "Does this chunk directly help answer the query? Answer: yes or no."
        )}],
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")


async def run_judge_batched(tasks, keys, batch_size=20):
    results = {}
    for i in range(0, len(tasks), batch_size):
        batch = await asyncio.gather(*tasks[i:i+batch_size], return_exceptions=True)
        for key, res in zip(keys[i:i+batch_size], batch):
            results[key] = res if isinstance(res, bool) else False
        if i + batch_size < len(tasks):
            await asyncio.sleep(3)
    return results


def rrf_merge(lists):
    scores = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


async def retrieve_parents(session_id, doc_idx, chunks, query, n=40):
    cap = min(n * 2, len(chunks))
    query_emb = await embed_query(query)
    dense_res = query_dense(session_id, doc_idx, query_emb, n_results=cap)
    dense_ids = [int(id_.split("_")[1]) for id_ in dense_res["ids"][0]]
    bm25 = build_bm25([c.text for c in chunks])
    sparse_ids = [idx for idx, _ in bm25_search(bm25, query, n=cap)]
    rrf = rrf_merge([dense_ids, sparse_ids])
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    seen, parent_ids, parent_texts = set(), [], []
    for child_id, _ in ranked:
        if child_id >= len(chunks): continue
        p = chunks[child_id].parent_index
        if p not in seen:
            seen.add(p)
            parent_ids.append(p)
            parent_texts.append(chunks[child_id].parent_text)
    return parent_ids, parent_texts


def recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


async def evaluate_doc(path: str, doc_idx: int):
    name = path.split("/")[-1].replace(".pdf", "")
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    with open(path, "rb") as f:
        result = parse_pdf(f.read(), path.split("/")[-1])
    raw_text = result["raw_text"]

    # Build baseline chunks to generate eval_set (queries are config-independent)
    base_chunks = build_chunks(raw_text, 400, 60)
    parent_map = {}
    for c in base_chunks:
        if c.parent_index not in parent_map:
            parent_map[c.parent_index] = c
    unique_parents = list(parent_map.values())

    random.seed(42)
    sampled = random.sample(unique_parents, min(SAMPLE_CHUNKS, len(unique_parents)))
    print(f"  Generating queries for {len(sampled)} chunks ({QUERIES_PER} per chunk)...")

    query_lists = await asyncio.gather(*[generate_queries(c.parent_text) for c in sampled])
    eval_set = [(q, c.parent_index) for c, qs in zip(sampled, query_lists) for q in qs]
    print(f"  {len(eval_set)} queries generated.")

    # ── Phase 1: embed + retrieve for both configs ──────────────────────────
    config_data = {}
    for cfg in CONFIGS:
        print(f"  [{cfg['name']}] embedding + retrieving...")
        chunks = build_chunks(raw_text, cfg["child_chars"], cfg["child_overlap"])
        embeddings = await embed_texts([c.text for c in chunks])
        delete_doc_collection(SESSION_ID, doc_idx)
        add_chunks(SESSION_ID, doc_idx, chunks, embeddings)

        cands = {}
        for query, src_pid in eval_set:
            parent_ids, parent_texts = await retrieve_parents(SESSION_ID, doc_idx, chunks, query)
            cands[(query, src_pid)] = (parent_ids, parent_texts)
        config_data[cfg["name"]] = {"chunks": chunks, "cands": cands}

    delete_doc_collection(SESSION_ID, doc_idx)

    # ── Phase 2: shared judge cache (parent chunks identical across configs) ─
    print(f"  Running LLM-as-judge (shared across both configs)...")
    judge_cache = {}
    judge_tasks, judge_keys = [], []
    for cfg_name, data in config_data.items():
        for (query, src_pid), (parent_ids, parent_texts) in data["cands"].items():
            for pid, ptext in zip(parent_ids[:20], parent_texts[:20]):
                key = (query, pid)
                if key not in judge_cache:
                    judge_cache[key] = None
                    judge_tasks.append(is_relevant(query, ptext))
                    judge_keys.append(key)
    # mark source chunks always relevant
    for query, src_pid in eval_set:
        judge_cache[(query, src_pid)] = True

    judge_results = await run_judge_batched(judge_tasks, judge_keys)
    judge_cache.update(judge_results)

    print(f"  {len(judge_tasks)} judge calls (shared, not doubled).")

    # ── Phase 3: compute recall per config ──────────────────────────────────
    print(f"\n  {'Config':<25} " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    print(f"  {'-'*52}")

    all_results = {}
    for cfg in CONFIGS:
        cands = config_data[cfg["name"]]["cands"]
        metrics = {k: [] for k in K_VALUES}
        for (query, src_pid), (parent_ids, _) in cands.items():
            relevant = {pid for pid in (set(parent_ids[:20]) | {src_pid})
                        if judge_cache.get((query, pid), False)}
            if not relevant: continue
            for k in K_VALUES:
                metrics[k].append(recall_at_k(parent_ids, relevant, k))
        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        res = {k: avg(metrics[k]) for k in K_VALUES}
        all_results[cfg["name"]] = res
        row = f"  {cfg['name']:<25} " + "  ".join(f"{res[k]:.3f}" for k in K_VALUES)
        print(row)

    return all_results


async def main():
    random.seed(42)
    print("=" * 60)
    print("  DecideIQ -- Chunking Experiment: 400-char vs 600-char")
    print(f"  {SAMPLE_CHUNKS} chunks x {QUERIES_PER} query/chunk x {len(FILES)} docs x 2 configs")
    print(f"  Judge cache shared across configs to halve API calls.")
    print("=" * 60)

    aggregate = {cfg["name"]: {k: [] for k in K_VALUES} for cfg in CONFIGS}
    for idx, path in enumerate(FILES):
        doc_results = await evaluate_doc(path, idx)
        for cfg_name, res in doc_results.items():
            for k in K_VALUES:
                aggregate[cfg_name][k].append(res[k])

    print(f"\n\n{'='*60}")
    print("  AGGREGATE RECALL (avg across all 3 insurance docs)")
    print(f"{'='*60}")
    print(f"  {'Config':<25} " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    print(f"  {'-'*52}")
    for cfg_name in aggregate:
        vals = aggregate[cfg_name]
        row = f"  {cfg_name:<25} " + "  ".join(
            f"{sum(vals[k])/len(vals[k]):.3f}" for k in K_VALUES
        )
        print(row)

    # Verdict
    print(f"\n  Verdict:")
    for k in K_VALUES:
        b = sum(aggregate[CONFIGS[0]["name"]][k]) / len(aggregate[CONFIGS[0]["name"]][k])
        e = sum(aggregate[CONFIGS[1]["name"]][k]) / len(aggregate[CONFIGS[1]["name"]][k])
        diff = e - b
        winner = "600-char" if diff > 0.01 else ("400-char" if diff < -0.01 else "tie")
        print(f"    R@{k}: 400={b:.3f}  600={e:.3f}  diff={diff:+.3f}  -> {winner}")
    print()


asyncio.run(main())
