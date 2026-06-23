"""
Layout chunking eval — compares character-based vs layout-aware (pymupdf4llm)
chunking on insurance PDFs using the same LLM-as-judge methodology.

Baseline R@8 (from test_chunking_experiment.py, 400-char):
  HDFC Ergo   : 0.616
  Niva Bupa   : 0.765
  Star Health : 0.798

Run: docker exec decideiq-backend-1 python tests/test_layout_chunking.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio, random, re
import fitz
import pymupdf4llm
from openai import AsyncOpenAI
from app.ingestion.pdf_parser import parse_pdf
from app.rag.chunker import build_parent_child_chunks, build_layout_chunks, _child_splitter
from app.rag.chunker import Chunk
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection, query_dense
from app.rag.bm25_store import build_bm25, bm25_search

FILES = [
    "/app/data/insurance/hdfc_ergo_optima_restore.pdf",
    "/app/data/insurance/niva_bupa_reassure_3.pdf",
    "/app/data/insurance/star_health_family_optima.pdf",
]

SESSION_ID    = "layout_chunking_eval"
SAMPLE_CHUNKS = 10
QUERIES_PER   = 1
RRF_K         = 60
K_VALUES      = [5, 8, 20]

BASELINE_R8 = {
    "hdfc_ergo_optima_restore": 0.616,
    "niva_bupa_reassure_3": 0.765,
    "star_health_family_optima": 0.798,
}

_oai = AsyncOpenAI()


def extract_md_sections(file_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc)
    doc.close()
    parts = re.split(r'\n(?=#{1,4} )', md)
    return [p.strip() for p in parts if p.strip()]


async def generate_queries(chunk_text: str) -> list[str]:
    resp = await _oai.chat.completions.create(
        model="gpt-4o-mini", temperature=0.3,
        messages=[{"role": "user", "content": (
            f"Write {QUERIES_PER} short specific question(s) whose answer is directly in this text. "
            "Output one question per line, no numbering.\n\nText:\n{chunk_text[:700]}"
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
        file_bytes = f.read()

    # Build both chunk sets
    with open(path, "rb") as f:
        parsed = parse_pdf(f.read(), path.split("/")[-1])
    raw_text = parsed["raw_text"]

    char_chunks = build_parent_child_chunks(raw_text)

    sections = extract_md_sections(file_bytes)
    layout_chunks = build_layout_chunks(sections)

    print(f"  Char chunks : {len(char_chunks)} children, "
          f"{len(set(c.parent_index for c in char_chunks))} parents")
    print(f"  Layout chunks: {len(layout_chunks)} children, "
          f"{len(set(c.parent_index for c in layout_chunks))} parents ({len(sections)} sections)")

    # Generate eval_set from char_chunks (same baseline as chunking experiment)
    char_parent_map = {}
    for c in char_chunks:
        if c.parent_index not in char_parent_map:
            char_parent_map[c.parent_index] = c
    sampled = random.sample(list(char_parent_map.values()), min(SAMPLE_CHUNKS, len(char_parent_map)))

    print(f"  Generating {len(sampled)} queries...")
    query_lists = await asyncio.gather(*[generate_queries(c.parent_text) for c in sampled])
    eval_set = [(q, c.parent_index) for c, qs in zip(sampled, query_lists) for q in qs]
    print(f"  {len(eval_set)} queries.")

    configs = [
        ("char-based (baseline)", char_chunks),
        ("layout-aware (new)", layout_chunks),
    ]

    # Retrieve candidates for both configs
    config_cands = {}
    for cfg_name, chunks in configs:
        print(f"  [{cfg_name}] embedding + retrieving...")
        embeddings = await embed_texts([c.text for c in chunks])
        delete_doc_collection(SESSION_ID, doc_idx)
        add_chunks(SESSION_ID, doc_idx, chunks, embeddings)
        cands = {}
        for query, src_pid in eval_set:
            parent_ids, parent_texts = await retrieve_parents(SESSION_ID, doc_idx, chunks, query)
            cands[(query, src_pid)] = (parent_ids, parent_texts)
        config_cands[cfg_name] = cands

    delete_doc_collection(SESSION_ID, doc_idx)

    # Shared judge cache
    print(f"  Running LLM-as-judge (shared)...")
    judge_cache = {}
    judge_tasks, judge_keys = [], []
    for cfg_name, cands in config_cands.items():
        for (query, src_pid), (parent_ids, parent_texts) in cands.items():
            for pid, ptext in zip(parent_ids[:20], parent_texts[:20]):
                key = (query, pid, cfg_name)
                if key not in judge_cache:
                    judge_cache[key] = None
                    judge_tasks.append(is_relevant(query, ptext))
                    judge_keys.append(key)
    for query, src_pid in eval_set:
        for cfg_name in config_cands:
            judge_cache[(query, src_pid, cfg_name)] = True

    results_map = await run_judge_batched(judge_tasks, judge_keys)
    judge_cache.update(results_map)
    print(f"  {len(judge_tasks)} judge calls.")

    # Compute recall
    print(f"\n  {'Config':<25} " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    print(f"  {'-'*52}")

    all_results = {}
    for cfg_name, cands in config_cands.items():
        metrics = {k: [] for k in K_VALUES}
        for (query, src_pid), (parent_ids, _) in cands.items():
            relevant = {pid for pid in (set(parent_ids[:20]) | {src_pid})
                        if judge_cache.get((query, pid, cfg_name), False)}
            if not relevant: continue
            for k in K_VALUES:
                metrics[k].append(recall_at_k(parent_ids, relevant, k))
        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        res = {k: avg(metrics[k]) for k in K_VALUES}
        all_results[cfg_name] = res
        baseline_r8 = BASELINE_R8.get(name, 0)
        delta = res[8] - baseline_r8
        row = f"  {cfg_name:<25} " + "  ".join(f"{res[k]:.3f}" for k in K_VALUES)
        if "layout" in cfg_name:
            row += f"   (R@8 vs baseline: {delta:+.3f})"
        print(row)

    return all_results


async def main():
    random.seed(42)
    print("=" * 60)
    print("  Layout Chunking Eval vs Character-Based Baseline")
    print("=" * 60)

    aggregate = {}
    for idx, path in enumerate(FILES):
        doc_results = await evaluate_doc(path, idx)
        for cfg_name, res in doc_results.items():
            if cfg_name not in aggregate:
                aggregate[cfg_name] = {k: [] for k in K_VALUES}
            for k in K_VALUES:
                aggregate[cfg_name][k].append(res[k])

    print(f"\n\n{'='*60}")
    print("  AGGREGATE (avg across 3 insurance docs)")
    print(f"{'='*60}")
    print(f"  {'Config':<25} " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    print(f"  {'-'*52}")
    for cfg_name, vals in aggregate.items():
        row = f"  {cfg_name:<25} " + "  ".join(
            f"{sum(vals[k])/len(vals[k]):.3f}" for k in K_VALUES
        )
        print(row)

    # Final verdict
    char_r8 = sum(aggregate["char-based (baseline)"][8]) / 3
    layout_r8 = sum(aggregate["layout-aware (new)"][8]) / 3
    diff = layout_r8 - char_r8
    print(f"\n  R@8 delta: {diff:+.3f}")
    if diff > 0.02:
        print("  VERDICT: Layout chunking IMPROVES retrieval — keep the change.")
    elif diff < -0.02:
        print("  VERDICT: Layout chunking HURTS retrieval — revert.")
    else:
        print("  VERDICT: No meaningful difference — keep char-based (simpler).")
    print()


asyncio.run(main())
