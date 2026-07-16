"""
Phase 6 test: full comparison engine on two job offer documents.
Run: docker exec decideiq-backend-1 python tests/test_comparison_phase6.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.agents.question_generator import generate_questions
from app.comparison.engine import run_comparison

SESSION_ID = "test_session_phase6"

FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp Offer"),
    ("/app/data/job_offers/offer_b_finedge.txt", "FinEdge Offer"),
]


async def setup_docs() -> list[dict]:
    docs = []
    for idx, (path, name) in enumerate(FILES):
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()

        word_count = len(raw_text.split())
        will_use_rag = word_count > 3000

        delete_doc_collection(SESSION_ID, idx)
        if will_use_rag:
            chunks = build_parent_child_chunks(raw_text)
            embeddings = await embed_texts([c.text for c in chunks])
            add_chunks(SESSION_ID, idx, chunks, embeddings)

        docs.append({
            "name": name,
            "raw_text": raw_text,
            "word_count": word_count,
            "will_use_rag": will_use_rag,
        })
        print(f"  [{idx}] {name} — {word_count} words, will_use_rag={will_use_rag}")

    return docs


async def main():
    print("Setting up documents...")
    docs = await setup_docs()

    print("\nGenerating questions...")
    all_questions: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        qs = await generate_questions(doc["raw_text"], doc["name"])
        for q in qs:
            if q.lower() not in seen:
                seen.add(q.lower())
                all_questions.append(q)
    print(f"  {len(all_questions)} unique questions generated")

    fake_session = {"documents": docs, "generated_questions": all_questions}

    print("\nRunning comparison engine...\n")
    result = await run_comparison(SESSION_ID, fake_session)

    print("=" * 60)
    for qr in result["question_results"]:
        print(f"\nQ: {qr['question']}")
        for pd in qr["per_doc"]:
            print(f"  [{pd['doc_name']}] score={pd['score']}/10 | grounding={pd['grounding_score']}")
            print(f"    {pd['answer'][:140]}")

    print("\n" + "=" * 60)
    print("SCORES:")
    for ds in result["doc_summaries"]:
        print(f"  {ds['doc_name']}: {ds['raw_score']} pts → {ds['percentage']}%")

    print(f"\nWINNER: {result['winner_name']}")

    for idx in range(len(docs)):
        delete_doc_collection(SESSION_ID, idx)
    print("\nTest collections cleaned up.")


asyncio.run(main())
