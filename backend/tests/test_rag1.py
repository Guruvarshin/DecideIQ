"""
Phase 3 test: chunking -> embedding -> ChromaDB -> BM25 -> RRF retrieval
Run inside container: docker exec decideiq-backend-1 python tests/test_rag_phase3.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts, embed_query
from app.rag.vector_store import add_chunks, query_dense, delete_doc_collection
from app.rag.bm25_store import build_bm25, bm25_search
from app.rag.retriever import retrieve

SESSION_ID = "test_session_phase3"
DOC_IDX = 0

with open("/app/data/job_offers/offer_a_techcorp.txt", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Document length: {len(raw_text.split())} words")


async def main():
    # --- Chunking ---
    chunks = build_parent_child_chunks(raw_text)
    print(f"\nChunks: {len(chunks)} children")
    parent_count = len(set(c.parent_index for c in chunks))
    print(f"Parents: {parent_count}")
    print(f"\nSample child[0]:\n  '{chunks[0].text[:120]}...'")
    print(f"Sample parent[0]:\n  '{chunks[0].parent_text[:200]}...'")

    # --- Embeddings ---
    print("\nEmbedding child chunks...")
    child_texts = [c.text for c in chunks]
    embeddings = await embed_texts(child_texts)
    print(f"Embeddings: {len(embeddings)} vectors, dim={len(embeddings[0])}")

    # --- ChromaDB store ---
    delete_doc_collection(SESSION_ID, DOC_IDX)
    add_chunks(SESSION_ID, DOC_IDX, chunks, embeddings)
    print(f"Stored in ChromaDB collection: s{SESSION_ID}_d{DOC_IDX}")

    # --- Dense retrieval ---
    query = "What is the salary and location of this job?"
    print(f"\nQuery: '{query}'")
    q_emb = await embed_query(query)
    dense_res = query_dense(SESSION_ID, DOC_IDX, q_emb, n_results=5)
    print("\nTop-5 dense child chunks:")
    for i, (doc, dist) in enumerate(zip(dense_res["documents"][0], dense_res["distances"][0])):
        print(f"  [{i+1}] dist={dist:.4f} | '{doc[:80]}...'")

    # --- BM25 retrieval ---
    bm25 = build_bm25(child_texts)
    sparse = bm25_search(bm25, query, n=5)
    print("\nTop-5 BM25 child chunks:")
    for rank, (idx, score) in enumerate(sparse):
        print(f"  [{rank+1}] score={score:.4f} | '{child_texts[idx][:80]}...'")

    # --- RRF hybrid retrieval (final parents) ---
    parents = await retrieve(SESSION_ID, DOC_IDX, raw_text, query, top_k=3)
    print(f"\nRRF top-3 parent contexts returned:")
    for i, p in enumerate(parents):
        print(f"\n  Parent [{i+1}] ({len(p.split())} words):\n  '{p[:200]}...'")

    delete_doc_collection(SESSION_ID, DOC_IDX)
    print("\nTest collection cleaned up.")


asyncio.run(main())
