"""
Phase 4 test: full advanced RAG pipeline on a job offer document.
Run: docker exec decideiq-backend-1 python tests/test_rag_phase4.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.rag.pipeline import run_pipeline

SESSION_ID = "test_session_phase4"
DOC_IDX = 0
QUESTION = "What is the salary, work location, and notice period for this job?"

with open("/app/data/job_offers/offer_a_techcorp.txt", encoding="utf-8") as f:
    raw_text = f.read()


async def main():
    # Set up ChromaDB (as document upload would)
    print("Setting up ChromaDB for test document...")
    delete_doc_collection(SESSION_ID, DOC_IDX)
    chunks = build_parent_child_chunks(raw_text)
    embeddings = await embed_texts([c.text for c in chunks])
    add_chunks(SESSION_ID, DOC_IDX, chunks, embeddings)
    print(f"Stored {len(chunks)} chunks.\n")

    print(f"Question: '{QUESTION}'\n")
    print("=" * 60)

    result = await run_pipeline(
        session_id=SESSION_ID,
        doc_idx=DOC_IDX,
        raw_text=raw_text,
        question=QUESTION,
        will_use_rag=True,
    )

    print(f"Source  : {result['source']}")
    print(f"Grounding score: {result['grounding_score']}")
    print(f"Contexts returned: {len(result['contexts'])}\n")
    for i, ctx in enumerate(result["contexts"]):
        print(f"--- Context [{i+1}] ---")
        print(ctx[:400])
        print()

    delete_doc_collection(SESSION_ID, DOC_IDX)
    print("Test collection cleaned up.")


asyncio.run(main())
