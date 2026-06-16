"""
Phase 9 test: new question generation (title + user questions),
CRAG not-mentioned fix, full 5-metric RAGAS eval.

Run: docker exec decideiq-backend-1 python tests/test_phase9.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.agents.question_generator import generate_questions
from app.comparison.engine import run_comparison
from app.evaluation.ragas_eval import build_ragas_dataset, run_ragas, run_ragas_full

SESSION_ID = "test_session_phase9"
TITLE = "Job offer comparison"
FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp Offer"),
    ("/app/data/job_offers/offer_b_finedge.txt", "FinEdge Offer"),
]
USER_QUESTIONS = [
    "What is the base salary?",
    "Is there remote work?",
    "What health benefits are provided?",
]


async def setup_and_compare():
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

    print(f"\nGenerating questions from title: '{TITLE}'")
    print(f"User-provided questions: {USER_QUESTIONS}")
    questions = await generate_questions(title=TITLE, user_questions=USER_QUESTIONS)
    print(f"\nFinal question set ({len(questions)} questions):")
    for i, q in enumerate(questions, 1):
        prefix = "[rephrased]" if i <= len(USER_QUESTIONS) else "[generated]"
        print(f"  {i}. {prefix} {q}")

    result = await run_comparison(SESSION_ID, {"documents": docs, "generated_questions": questions})
    return docs, result


def check_not_mentioned(result):
    print("\n--- CRAG / Not-Mentioned Check ---")
    for qr in result["question_results"]:
        for pd in qr["per_doc"]:
            if pd["source"] == "not_mentioned":
                print(f"  ✓ '{qr['question'][:60]}...' → {pd['doc_name']}: NOT MENTIONED (no hallucination)")


def evaluate_production(doc_name, question_results):
    questions, answers, contexts = [], [], []
    for qr in question_results:
        pd = next((p for p in qr["per_doc"] if p["doc_name"] == doc_name), None)
        if pd and pd["answer"] not in ("Not found in document.", "Not mentioned in available sources."):
            questions.append(qr["question"])
            answers.append(pd["answer"])
            ctx = pd.get("contexts", [])
            contexts.append(ctx if ctx else [pd["answer"]])
    if not questions:
        return None, []
    dataset = build_ragas_dataset(questions, answers, contexts)
    return run_ragas(dataset), questions


def evaluate_full(doc_name, question_results, golden_answers):
    questions, answers, contexts = [], [], []
    for qr in question_results:
        pd = next((p for p in qr["per_doc"] if p["doc_name"] == doc_name), None)
        if pd and pd["answer"] not in ("Not found in document.", "Not mentioned in available sources."):
            questions.append(qr["question"])
            answers.append(pd["answer"])
            ctx = pd.get("contexts", [])
            contexts.append(ctx if ctx else [pd["answer"]])
    if not questions or len(golden_answers) != len(questions):
        return None
    dataset = build_ragas_dataset(questions, answers, contexts, references=golden_answers)
    return run_ragas_full(dataset)


async def main():
    print("=" * 60)
    print("PHASE 9 TEST")
    print("=" * 60)

    docs, result = await setup_and_compare()

    print("\n--- Comparison Scores ---")
    for ds in result["doc_summaries"]:
        print(f"  {ds['doc_name']}: {ds['percentage']}%")
    print(f"  Winner: {result['winner_name']}")

    check_not_mentioned(result)

    print("\n--- Production RAGAS Eval (3 metrics, no golden) ---")
    for doc in docs:
        metrics, questions = evaluate_production(doc["name"], result["question_results"])
        if not metrics:
            print(f"  {doc['name']}: No answerable questions.")
            continue
        print(f"\n  {doc['name']} ({metrics['n_questions'] if 'n_questions' in metrics else len(questions)} questions):")
        print(f"    Faithfulness        : {metrics.get('faithfulness', 'N/A')}")
        print(f"    Answer Relevancy    : {metrics.get('answer_relevancy', 'N/A')}")
        print(f"    Confidence Score    : {metrics['confidence_score']}")

    # Full 5-metric eval requires golden answers — using manually written ones for TechCorp
    # In production, these would be provided by the user via POST /evaluate/full
    print("\n--- Full 5-metric RAGAS Eval (TechCorp, with golden answers) ---")
    techcorp_result = None
    for qr in result["question_results"]:
        for p in qr["per_doc"]:
            if p["doc_name"] == "TechCorp Offer":
                techcorp_result = True
                break
        if techcorp_result:
            break

    # Count answerable questions for TechCorp
    tc_questions = [
        qr for qr in result["question_results"]
        if any(
            p["doc_name"] == "TechCorp Offer" and
            p["answer"] not in ("Not found in document.", "Not mentioned in available sources.")
            for p in qr["per_doc"]
        )
    ]
    n = len(tc_questions)
    # Write one golden answer per answerable question (generic placeholders for test)
    golden = [f"The answer to '{q['question'][:40]}' can be found in the TechCorp offer document." for q in tc_questions]

    if n > 0:
        full_metrics = evaluate_full("TechCorp Offer", result["question_results"], golden)
        if full_metrics:
            print(f"  Faithfulness        : {full_metrics.get('faithfulness', 'N/A')}")
            print(f"  Answer Relevancy    : {full_metrics.get('answer_relevancy', 'N/A')}")
            print(f"  Context Precision   : {full_metrics.get('context_precision', 'N/A')}")
            print(f"  Context Recall      : {full_metrics.get('context_recall', 'N/A')}")
            print(f"  Answer Correctness  : {full_metrics.get('answer_correctness', 'N/A')}")
            print(f"  Confidence Score    : {full_metrics['confidence_score']}")
        else:
            print("  Could not run full eval (golden answer count mismatch).")

    for idx in range(len(docs)):
        delete_doc_collection(SESSION_ID, idx)
    print("\nDone.")


asyncio.run(main())
