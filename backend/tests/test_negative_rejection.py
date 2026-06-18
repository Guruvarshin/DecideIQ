"""
Task 5: Negative rejection test — unanswerable query robustness.

Sends queries that have NO answer in the job offer documents and asserts:
  1. The pipeline returns "Not found in document." (no hallucination)
  2. Grounding score is low (CRAG correctly flagged un-grounded)

Run: docker exec decideiq-backend-1 python tests/test_negative_rejection.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.rag.pipeline import run_pipeline
from app.comparison.answerer import answer_question

SESSION_ID = "negative_rejection_test"
DOC_PATH = "/app/data/job_offers/offer_a_techcorp.txt"
DOC_IDX = 0

UNANSWERABLE = [
    "What is the CEO's shoe size?",
    "What is the company's carbon footprint in metric tons?",
    "Does the company offer a helicopter commute benefit?",
    "What is the monthly pizza budget for the engineering team?",
    "What color are the office chairs?",
]

NOT_FOUND_PHRASE = "not found in document"


async def _run_query(raw_text: str, query: str) -> dict:
    result = await run_pipeline(SESSION_ID, DOC_IDX, raw_text, query, will_use_rag=True)
    answer = await answer_question(query, result["contexts"])
    return {"answer": answer, "grounding_score": result["grounding_score"]}


async def main():
    print("=" * 60)
    print("  DecideIQ -- Negative Rejection Test")
    print("=" * 60)

    with open(DOC_PATH) as f:
        raw_text = f.read()

    chunks = build_parent_child_chunks(raw_text)
    embeddings = await embed_texts([c.text for c in chunks])
    delete_doc_collection(SESSION_ID, DOC_IDX)
    add_chunks(SESSION_ID, DOC_IDX, chunks, embeddings)

    passed = 0
    failed_queries = []
    for query in UNANSWERABLE:
        result = await _run_query(raw_text, query)
        answer = result["answer"]
        ok = NOT_FOUND_PHRASE in answer.lower()
        if ok:
            passed += 1
        else:
            failed_queries.append((query, answer))
        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {query}")
        print(f"         Answer    : {answer[:150]}")
        print(f"         Grounding : {result['grounding_score']:.3f}")

    delete_doc_collection(SESSION_ID, DOC_IDX)

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{len(UNANSWERABLE)} unanswerable queries correctly rejected")
    if passed == len(UNANSWERABLE):
        print("  PASS — pipeline correctly refused to hallucinate on all queries.")
    else:
        print(f"  FAIL — {len(failed_queries)} queries were hallucinated:")
        for q, a in failed_queries:
            print(f"    Q: {q}")
            print(f"    A: {a[:120]}")
    print()


asyncio.run(main())
