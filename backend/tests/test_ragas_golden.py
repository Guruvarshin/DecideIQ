"""
Full 5-metric RAGAS benchmark using a hand-written golden dataset.
Runs fixed questions through the RAG pipeline to get real (answer, contexts),
then evaluates against golden answers — making context_recall and
answer_correctness meaningful.

Run: docker exec decideiq-backend-1 python tests/test_ragas_golden.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection
from app.comparison.answerer import answer_question
from app.rag.pipeline import run_pipeline
from app.evaluation.ragas_eval import build_ragas_dataset, run_ragas, run_ragas_full

SESSION_ID = "golden_eval_session"
FILES = [
    "/app/data/job_offers/offer_a_techcorp.txt",
    "/app/data/job_offers/offer_b_finedge.txt",
]

# ── Fixed questions — same set evaluated against both docs ────────────────────
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

# ── Golden answers grounded in offer_a_techcorp.txt ───────────────────────────
TECHCORP_GOLDEN = [
    "The annual base salary is ₹18,00,000 per annum, paid as ₹1,50,000 per month.",

    "Yes, remote work is permitted. The role follows a hybrid model: 3 days in office "
    "(Tuesday, Wednesday, Thursday) and 2 days remote per week.",

    "Health insurance is a ₹5,00,000 family floater covering self, spouse, and 2 children "
    "under Star Health with no waiting period. The company also provides term life insurance "
    "of ₹50,00,000 coverage at no cost to the employee.",

    "Base salary is ₹18,00,000 per annum. Performance bonus is up to 15% of annual base "
    "salary (up to ₹2,70,000), paid annually. A one-time joining bonus of ₹1,00,000 is "
    "provided with a 12-month clawback clause. 800 ESOP units vest over 4 years with a 25% "
    "cliff at 1 year and 6.25% quarterly thereafter, at a strike price of ₹120 per unit.",

    "Annual paid leave is 18 days. Sick leave is 12 days (non-encashable). Casual leave is "
    "6 days. Maternity leave is 26 weeks as per the Maternity Benefit Act. Paternity leave "
    "is 5 days.",

    "Provident Fund with 12% employer contribution on basic salary, amounting to ₹6,480 per "
    "month. Gratuity is provided as per the Payment of Gratuity Act, 1972.",

    "An annual learning budget of ₹25,000 is provided for courses, conferences, or "
    "certifications. An internet allowance of ₹1,500 per month and a meal card of ₹2,200 "
    "per month (tax-exempt) are also provided.",

    "Working hours are flexible with core hours from 11:00 AM to 4:00 PM IST. The hybrid "
    "model requires office presence on Tuesday, Wednesday, and Thursday. No overtime or "
    "weekend work requirements are mentioned in the offer.",
]

# ── Golden answers grounded in offer_b_finedge.txt ────────────────────────────
FINEDGE_GOLDEN = [
    "The basic salary is ₹8,16,000 per annum (₹68,000 per month). The total fixed CTC is "
    "₹20,40,000 per annum, which includes basic salary, HRA of ₹4,08,000, special allowance "
    "of ₹4,90,800, and LTA of ₹68,000.",

    "No, remote work is not permitted. The role requires full-time office presence, Monday to "
    "Friday, at the BKC Mumbai office.",

    "Health insurance provides ₹3,00,000 individual coverage under Niva Bupa. Family members "
    "can be added at ₹8,000 per annum per member, at the employee's cost. Term life insurance "
    "is not provided by the company, but employees may enrol in a group term scheme at ₹4,500 "
    "per annum for ₹50,00,000 coverage.",

    "Fixed CTC is ₹20,40,000 per annum. Variable pay is 20% of fixed CTC (₹4,08,000), split "
    "60% on individual performance and 40% on business unit performance, paid quarterly. There "
    "is no joining bonus. ESOPs are not applicable at this grade; eligibility is reviewed at "
    "Senior Engineer level after 18–24 months.",

    "Annual earned leave is 15 days. Sick leave is 8 days. Casual leave is 8 days. Maternity "
    "leave is 26 weeks as per statute. Paternity leave is 3 days.",

    "Provident Fund with 12% employer contribution on basic salary, amounting to ₹8,160 per "
    "month. This employer PF contribution is not included in the stated fixed CTC. Gratuity "
    "is provided as per statute.",

    "No learning budget or internet allowance is provided. The offer makes no mention of any "
    "professional development or training support.",

    "Fixed hours are 9:30 AM to 6:30 PM, Monday to Friday, full-time office. Occasional "
    "weekend work may be required during release cycles, compensated with compensatory off "
    "rather than additional pay.",
]


async def setup_doc(path: str, idx: int) -> dict:
    with open(path, encoding="utf-8") as f:
        raw_text = f.read()
    delete_doc_collection(SESSION_ID, idx)
    chunks = build_parent_child_chunks(raw_text)
    embeddings = await embed_texts([c.text for c in chunks])
    add_chunks(SESSION_ID, idx, chunks, embeddings)
    return {"raw_text": raw_text, "will_use_rag": True}


async def run_all_questions(doc: dict, doc_idx: int) -> tuple[list[str], list[list[str]]]:
    answers, contexts_per_q = [], []
    for question in QUESTIONS:
        result = await run_pipeline(
            session_id=SESSION_ID,
            doc_idx=doc_idx,
            raw_text=doc["raw_text"],
            question=question,
            will_use_rag=True,
        )
        answer = await answer_question(question, result["contexts"])
        answers.append(answer)
        contexts_per_q.append(result["contexts"])
    return answers, contexts_per_q


def print_results(label: str, metrics: dict) -> None:
    bar = "=" * 58
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    for key in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
        val = metrics.get(key)
        display = f"{val:.4f}" if isinstance(val, float) else "N/A"
        print(f"  {key.replace('_', ' ').title():<24}: {display}")
    print(f"  {'Confidence Score':<24}: {metrics['confidence_score']:.4f}")


def print_per_question(metrics: dict, golden: list[str]) -> None:
    print("\n  Per-question breakdown:")
    for i, pq in enumerate(metrics["per_question"]):
        faith = pq.get("faithfulness", "?")
        rel   = pq.get("answer_relevancy", "?")
        rec   = pq.get("context_recall", "?")
        corr  = pq.get("answer_correctness", "?")
        print(f"\n  Q{i+1}: {QUESTIONS[i]}")
        print(f"    Golden : {golden[i][:90]}")
        faith_s = f"{faith:.3f}" if isinstance(faith, float) else faith
        rel_s   = f"{rel:.3f}"   if isinstance(rel, float)   else rel
        rec_s   = f"{rec:.3f}"   if isinstance(rec, float)   else rec
        corr_s  = f"{corr:.3f}"  if isinstance(corr, float)  else corr
        print(f"    faith={faith_s}  rel={rel_s}  recall={rec_s}  correctness={corr_s}")


async def main():
    names   = ["TechCorp Offer", "FinEdge Offer"]
    goldens = [TECHCORP_GOLDEN, FINEDGE_GOLDEN]

    print("=" * 58)
    print("  DecideIQ — Full RAGAS Golden Benchmark")
    print("=" * 58)
    print(f"\n{len(QUESTIONS)} fixed questions evaluated against both documents.\n")

    for idx, (path, name, golden) in enumerate(zip(FILES, names, goldens)):
        print(f"\n{'─' * 58}")
        print(f"  Setting up {name}...")
        doc = await setup_doc(path, idx)

        print(f"  Running {len(QUESTIONS)} questions through RAG pipeline...")
        answers, contexts_per_q = await run_all_questions(doc, idx)

        # Production eval — no golden answers needed
        prod_ds = build_ragas_dataset(QUESTIONS, answers, contexts_per_q)
        prod_metrics = run_ragas(prod_ds)
        print_results(f"{name} — Production (faithfulness + answer_relevancy)", prod_metrics)

        # Full eval — with golden answers
        full_ds = build_ragas_dataset(QUESTIONS, answers, contexts_per_q, references=golden)
        full_metrics = run_ragas_full(full_ds)
        print_results(f"{name} — Full 5-metric (with golden answers)", full_metrics)
        print_per_question(full_metrics, golden)

    for idx in range(len(FILES)):
        delete_doc_collection(SESSION_ID, idx)

    print("\n\nDone.")


asyncio.run(main())
