"""
Phase 7 test: full pipeline ending with Claude Sonnet verdict.
Run: docker exec decideiq-backend-1 python tests/test_verdict_phase7.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.agents.question_generator import generate_questions
from app.comparison.engine import run_comparison
from app.comparison.verdict import generate_verdict

SESSION_ID = "test_session_phase7"
FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp Offer"),
    ("/app/data/job_offers/offer_b_finedge.txt", "FinEdge Offer"),
]


async def setup_docs() -> list[dict]:
    docs = []
    for idx, (path, name) in enumerate(FILES):
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()
        will_use_rag = len(raw_text.split()) > 3000
        delete_doc_collection(SESSION_ID, idx)
        if will_use_rag:
            chunks = build_parent_child_chunks(raw_text)
            embeddings = await embed_texts([c.text for c in chunks])
            add_chunks(SESSION_ID, idx, chunks, embeddings)
        docs.append({"name": name, "raw_text": raw_text, "will_use_rag": will_use_rag})
    return docs


async def main():
    print("Setting up documents...")
    docs = await setup_docs()

    print("Generating questions...")
    seen: set[str] = set()
    questions: list[str] = []
    for doc in docs:
        for q in await generate_questions(doc["raw_text"], doc["name"]):
            if q.lower() not in seen:
                seen.add(q.lower())
                questions.append(q)
    print(f"  {len(questions)} questions\n")

    print("Running comparison engine...")
    result = await run_comparison(SESSION_ID, {"documents": docs, "generated_questions": questions})

    print("Generating verdict (Claude Sonnet)...\n")
    verdict = await generate_verdict(result)

    print("=" * 60)
    print("SCORES")
    print("=" * 60)
    for ds in result["doc_summaries"]:
        print(f"  {ds['doc_name']}: {ds['raw_score']} pts → {ds['percentage']}%")
    print(f"\nWINNER: {result['winner_name']}\n")

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(verdict)

    for idx in range(len(docs)):
        delete_doc_collection(SESSION_ID, idx)


asyncio.run(main())
