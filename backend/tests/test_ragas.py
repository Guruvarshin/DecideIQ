"""
Phase 8 test: RAGAS evaluation on comparison output.
Run: docker exec decideiq-backend-1 python tests/test_ragas_phase8.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.agents.question_generator import generate_questions
from app.comparison.engine import run_comparison
from app.evaluation.ragas_eval import build_ragas_dataset, run_ragas

SESSION_ID = "test_session_phase8"
FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp Offer"),
    ("/app/data/job_offers/offer_b_finedge.txt", "FinEdge Offer"),
]


async def setup_and_compare() -> tuple[list[dict], dict]:
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

    seen: set[str] = set()
    questions: list[str] = []
    for doc in docs:
        for q in await generate_questions(doc["raw_text"], doc["name"]):
            if q.lower() not in seen:
                seen.add(q.lower())
                questions.append(q)

    result = await run_comparison(SESSION_ID, {"documents": docs, "generated_questions": questions})
    return docs, result


def evaluate_doc(doc_name: str, question_results: list[dict]) -> dict:
    questions, answers, contexts = [], [], []
    for qr in question_results:
        pd = next((p for p in qr["per_doc"] if p["doc_name"] == doc_name), None)
        if pd and pd["answer"] != "Not found in document.":
            questions.append(qr["question"])
            answers.append(pd["answer"])
            contexts.append([pd["answer"]])
    if not questions:
        return {}
    dataset = build_ragas_dataset(questions, answers, contexts)
    return run_ragas(dataset)


async def main():
    print("Running full pipeline...")
    docs, result = await setup_and_compare()

    print("\n" + "=" * 60)
    print("COMPARISON SCORES")
    print("=" * 60)
    for ds in result["doc_summaries"]:
        print(f"  {ds['doc_name']}: {ds['percentage']}%")
    print(f"  Winner: {result['winner_name']}\n")

    for idx, doc in enumerate(docs):
        print(f"{'=' * 60}")
        print(f"RAGAS EVAL — {doc['name']}")
        print(f"{'=' * 60}")
        metrics = evaluate_doc(doc["name"], result["question_results"])
        if not metrics:
            print("  No answerable questions.\n")
            continue
        print(f"  Faithfulness      : {metrics['faithfulness']}")
        print(f"  Answer Relevancy  : {metrics['answer_relevancy']}")
        print(f"  Confidence Score  : {metrics['confidence_score']}")
        print(f"  (over {len(metrics['per_question'])} questions)\n")
        print("  Per-question breakdown:")
        for pq in metrics["per_question"]:
            print(f"    faith={pq['faithfulness']} rel={pq['answer_relevancy']}")
            print(f"    Q: {pq['question'][:80]}")

    for idx in range(len(docs)):
        delete_doc_collection(SESSION_ID, idx)
    print("\nDone.")


asyncio.run(main())
