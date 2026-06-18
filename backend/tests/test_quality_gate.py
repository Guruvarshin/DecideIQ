"""
Task 4: Quality gate — CI/CD regression check.

Runs 3 questions through the RAG pipeline (TechCorp offer) and asserts
minimum RAGAS thresholds. Fails with exit code 1 if the pipeline regresses.

Run standalone : docker exec decideiq-backend-1 python tests/test_quality_gate.py
Run via pytest : docker exec decideiq-backend-1 python -m pytest tests/test_quality_gate.py -v
  (requires: pip install pytest)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
from datasets import Dataset
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.rag.pipeline import run_pipeline
from app.comparison.answerer import answer_question

SESSION_ID = "quality_gate_session"
DOC_IDX = 0
DOC_PATH = "/app/data/job_offers/offer_a_techcorp.txt"

GATE_QUESTIONS = [
    "What is the annual base salary offered for this position?",
    "What specific health benefits are included in the employment package?",
    "What is the leave policy, including the number of paid days off per year?",
]

MIN_FAITHFULNESS     = 0.80
MIN_ANSWER_RELEVANCY = 0.65


async def run_eval() -> dict:
    with open(DOC_PATH) as f:
        raw_text = f.read()

    chunks = build_parent_child_chunks(raw_text)
    embeddings = await embed_texts([c.text for c in chunks])
    delete_doc_collection(SESSION_ID, DOC_IDX)
    add_chunks(SESSION_ID, DOC_IDX, chunks, embeddings)

    rows = []
    for q in GATE_QUESTIONS:
        result = await run_pipeline(SESSION_ID, DOC_IDX, raw_text, q, will_use_rag=True)
        answer = await answer_question(q, result["contexts"])
        rows.append({
            "question": q,
            "answer": answer,
            "contexts": result["contexts"],
            "ground_truth": "",
        })

    delete_doc_collection(SESSION_ID, DOC_IDX)

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        run_config=RunConfig(max_workers=1, max_retries=5, timeout=120),
    )
    def _scalar(v):
        # RAGAS 0.2.x returns a list of per-row scores; average them
        if isinstance(v, (list, tuple)):
            return sum(v) / len(v) if v else 0.0
        return float(v)

    return {
        "faithfulness": _scalar(result["faithfulness"]),
        "answer_relevancy": _scalar(result["answer_relevancy"]),
    }


def main():
    print("=" * 60)
    print("  DecideIQ -- Quality Gate")
    print("=" * 60)
    scores = asyncio.run(run_eval())

    faith = scores["faithfulness"]
    rel   = scores["answer_relevancy"]

    print(f"\n  Faithfulness     : {faith:.4f}  (min={MIN_FAITHFULNESS})")
    print(f"  Answer Relevancy : {rel:.4f}  (min={MIN_ANSWER_RELEVANCY})")

    failures = []
    if faith < MIN_FAITHFULNESS:
        failures.append(f"Faithfulness {faith:.4f} < {MIN_FAITHFULNESS}")
    if rel < MIN_ANSWER_RELEVANCY:
        failures.append(f"Answer relevancy {rel:.4f} < {MIN_ANSWER_RELEVANCY}")

    print()
    if not failures:
        print("  PASS — pipeline meets quality thresholds.")
        sys.exit(0)
    else:
        print("  FAIL:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
