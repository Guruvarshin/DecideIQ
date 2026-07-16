"""
Task 2: top_k grid search.

Hypothesis: retrieval is near-perfect (Recall=1.0 on short docs).
Bottleneck is noise from k=8 chunks passed to the LLM answerer.
Smaller k may improve answer_correctness by reducing irrelevant context.

Tests k = 3, 5, 8 against the 8-question golden dataset.
Measures answer_correctness and faithfulness via RAGAS full eval.

Run: docker exec decideiq-backend-1 python tests/test_topk_grid.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.comparison.answerer import answer_question
from app.rag.pipeline import run_pipeline
from app.evaluation.ragas_eval import build_ragas_dataset, run_ragas_full

SESSION_ID = "topk_grid_session"

FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp", 0),
    ("/app/data/job_offers/offer_b_finedge.txt",  "FinEdge",  1),
]

QUESTIONS = [
    "What is the annual base salary offered for this position?",
    "Is remote work permitted for this role, and what are the specific arrangements?",
    "What specific health benefits are included in the employment package?",
    "What is the total annual compensation including bonuses and stock options?",
    "What is the leave policy, including the number of paid days off per year?",
    "What is the retirement plan offered and what is the employer contribution?",
    "What opportunities for professional development and training are provided?",
    "What is the expected work schedule, and are there overtime or weekend work requirements?",
]

TECHCORP_GOLDEN = [
    "The annual base salary is Rs.18,00,000 per annum, paid as Rs.1,50,000 per month.",
    "Yes, remote work is permitted. The role follows a hybrid model: 3 days in office (Tuesday, Wednesday, Thursday) and 2 days remote per week.",
    "Health insurance is a Rs.5,00,000 family floater covering self, spouse, and 2 children under Star Health with no waiting period. The company also provides term life insurance of Rs.50,00,000 coverage at no cost to the employee.",
    "Base salary is Rs.18,00,000 per annum. Performance bonus is up to 15% of annual base salary (up to Rs.2,70,000), paid annually. A one-time joining bonus of Rs.1,00,000 is provided with a 12-month clawback clause. 800 ESOP units vest over 4 years with a 25% cliff at 1 year and 6.25% quarterly thereafter, at a strike price of Rs.120 per unit.",
    "Annual paid leave is 18 days. Sick leave is 12 days (non-encashable). Casual leave is 6 days. Maternity leave is 26 weeks as per the Maternity Benefit Act. Paternity leave is 5 days.",
    "Provident Fund with 12% employer contribution on basic salary, amounting to Rs.6,480 per month. Gratuity is provided as per the Payment of Gratuity Act, 1972.",
    "An annual learning budget of Rs.25,000 is provided for courses, conferences, or certifications. An internet allowance of Rs.1,500 per month and a meal card of Rs.2,200 per month (tax-exempt) are also provided.",
    "Working hours are flexible with core hours from 11:00 AM to 4:00 PM IST. The hybrid model requires office presence on Tuesday, Wednesday, and Thursday. No overtime or weekend work requirements are mentioned in the offer.",
]

FINEDGE_GOLDEN = [
    "The basic salary is Rs.8,16,000 per annum (Rs.68,000 per month). The total fixed CTC is Rs.20,40,000 per annum, which includes basic salary, HRA of Rs.4,08,000, special allowance of Rs.4,90,800, and LTA of Rs.68,000.",
    "No, remote work is not permitted. The role requires full-time office presence, Monday to Friday, at the BKC Mumbai office.",
    "Health insurance provides Rs.3,00,000 individual coverage under Niva Bupa. Family members can be added at Rs.8,000 per annum per member, at the employee's cost. Term life insurance is not provided by the company, but employees may enrol in a group term scheme at Rs.4,500 per annum for Rs.50,00,000 coverage.",
    "Fixed CTC is Rs.20,40,000 per annum. Variable pay is 20% of fixed CTC (Rs.4,08,000), split 60% on individual performance and 40% on business unit performance, paid quarterly. There is no joining bonus. ESOPs are not applicable at this grade; eligibility is reviewed at Senior Engineer level after 18-24 months.",
    "Annual earned leave is 15 days. Sick leave is 8 days. Casual leave is 8 days. Maternity leave is 26 weeks as per statute. Paternity leave is 3 days.",
    "Provident Fund with 12% employer contribution on basic salary, amounting to Rs.8,160 per month. This employer PF contribution is not included in the stated fixed CTC. Gratuity is provided as per statute.",
    "No learning budget or internet allowance is provided. The offer makes no mention of any professional development or training support.",
    "Fixed hours are 9:30 AM to 6:30 PM, Monday to Friday, full-time office. Occasional weekend work may be required during release cycles, compensated with compensatory off rather than additional pay.",
]

GOLDENS = [TECHCORP_GOLDEN, FINEDGE_GOLDEN]
TOP_K_VALUES = [3, 5, 8]


async def setup_doc(path: str, idx: int) -> dict:
    with open(path, encoding="utf-8") as f:
        raw_text = f.read()
    delete_doc_collection(SESSION_ID, idx)
    chunks = build_parent_child_chunks(raw_text)
    embeddings = await embed_texts([c.text for c in chunks])
    add_chunks(SESSION_ID, idx, chunks, embeddings)
    return {"raw_text": raw_text, "will_use_rag": True}


async def run_questions_at_k(doc: dict, doc_idx: int, top_k: int):
    answers, contexts_per_q = [], []
    for question in QUESTIONS:
        result = await run_pipeline(
            session_id=SESSION_ID,
            doc_idx=doc_idx,
            raw_text=doc["raw_text"],
            question=question,
            will_use_rag=True,
            top_k=top_k,
        )
        answer = await answer_question(question, result["contexts"])
        answers.append(answer)
        contexts_per_q.append(result["contexts"])
    return answers, contexts_per_q


async def main():
    print("=" * 60)
    print("  DecideIQ -- top_k Grid Search (Task 2)")
    print("=" * 60)
    print(f"  Testing top_k = {TOP_K_VALUES} on {len(QUESTIONS)} golden questions\n")

    all_results = {}  # {label: {k: metrics}}

    for path, label, idx in FILES:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        doc = await setup_doc(path, idx)
        all_results[label] = {}

        for k in TOP_K_VALUES:
            print(f"\n  Running top_k={k}...")
            answers, contexts_per_q = await run_questions_at_k(doc, idx, k)
            golden = GOLDENS[idx]
            ds = build_ragas_dataset(QUESTIONS, answers, contexts_per_q, references=golden)
            metrics = run_ragas_full(ds)
            all_results[label][k] = metrics

            print(f"  top_k={k}  faith={metrics.get('faithfulness', 0):.4f}  "
                  f"rel={metrics.get('answer_relevancy', 0):.4f}  "
                  f"recall={metrics.get('context_recall', 0):.4f}  "
                  f"correctness={metrics.get('answer_correctness', 0):.4f}  "
                  f"confidence={metrics['confidence_score']:.4f}")

        delete_doc_collection(SESSION_ID, idx)

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY -- Answer Correctness by top_k")
    print(f"{'='*60}")
    print(f"  {'Doc':<12} {'k=3':>8} {'k=5':>8} {'k=8':>8}  Best k")
    print(f"  {'-'*48}")
    for label in all_results:
        scores = {k: all_results[label][k].get('answer_correctness', 0) for k in TOP_K_VALUES}
        best_k = max(scores, key=scores.get)
        row = f"  {label:<12}"
        for k in TOP_K_VALUES:
            marker = " *" if k == best_k else "  "
            row += f" {scores[k]:>6.4f}{marker}"
        row += f"  k={best_k}"
        print(row)

    print(f"\n  {'Doc':<12} {'k=3':>8} {'k=5':>8} {'k=8':>8}  Best k  (Faithfulness)")
    print(f"  {'-'*48}")
    for label in all_results:
        scores = {k: all_results[label][k].get('faithfulness', 0) for k in TOP_K_VALUES}
        best_k = max(scores, key=scores.get)
        row = f"  {label:<12}"
        for k in TOP_K_VALUES:
            marker = " *" if k == best_k else "  "
            row += f" {scores[k]:>6.4f}{marker}"
        row += f"  k={best_k}"
        print(row)

    print("\nDone.")


asyncio.run(main())
