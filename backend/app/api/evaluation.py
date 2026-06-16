from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from bson import ObjectId
from app.core.database import get_database
from app.core.dependencies import get_current_user
from app.evaluation.ragas_eval import build_ragas_dataset, run_ragas, run_ragas_full

router = APIRouter(prefix="/sessions", tags=["evaluation"])

_NOT_MENTIONED = "Not mentioned in available sources."


def _extract_qa_triples(question_results: list[dict], doc_name: str):
    """Pull (question, answer, contexts) per doc, skip unanswerable rows."""
    questions, answers, contexts = [], [], []
    for qr in question_results:
        pd = next((p for p in qr["per_doc"] if p["doc_name"] == doc_name), None)
        if pd is None:
            continue
        answer = pd["answer"]
        # Skip rows where the pipeline found nothing useful
        if answer in ("Not found in document.", _NOT_MENTIONED):
            continue
        questions.append(qr["question"])
        answers.append(answer)
        # Use real retrieved contexts; fall back to the answer itself if missing
        ctx = pd.get("contexts", [])
        contexts.append(ctx if ctx else [answer])
    return questions, answers, contexts


@router.post("/{session_id}/evaluate", status_code=200)
async def evaluate_session(
    session_id: str,
    doc_idx: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """
    Production RAGAS eval — no golden answers needed.
    Metrics: faithfulness, answer_relevancy, context_precision.
    Confidence score = (faithfulness + answer_relevancy) / 2.
    """
    db = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session = await db.sessions.find_one({"_id": oid, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    comparison = session.get("comparison_result")
    if not comparison:
        raise HTTPException(status_code=400, detail="Run comparison first")

    docs = session.get("documents", [])
    if doc_idx >= len(docs):
        raise HTTPException(status_code=400, detail="doc_idx out of range")

    doc_name = docs[doc_idx]["name"]
    questions, answers, contexts = _extract_qa_triples(
        comparison.get("question_results", []), doc_name
    )

    if not questions:
        raise HTTPException(
            status_code=400, detail="No answerable questions found for this document"
        )

    dataset = build_ragas_dataset(questions, answers, contexts)
    metrics = await run_in_threadpool(run_ragas, dataset)
    metrics["doc_name"] = doc_name
    metrics["n_questions"] = len(questions)
    metrics["eval_type"] = "production"

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {f"ragas_eval.doc_{doc_idx}": metrics}},
    )
    return metrics


class _FullEvalBody(BaseModel):
    doc_idx: int = 0
    golden_answers: list[str]  # one per generated question, in order


@router.post("/{session_id}/evaluate/full", status_code=200)
async def evaluate_session_full(
    session_id: str,
    body: _FullEvalBody,
    current_user: dict = Depends(get_current_user),
):
    """
    Full 5-metric RAGAS eval. Requires golden answers (reference) provided by caller.
    Metrics: faithfulness, answer_relevancy, context_precision,
             context_recall, answer_correctness.
    Use for development benchmarking and portfolio-quality RAGAS reports.
    """
    db = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session = await db.sessions.find_one({"_id": oid, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    comparison = session.get("comparison_result")
    if not comparison:
        raise HTTPException(status_code=400, detail="Run comparison first")

    docs = session.get("documents", [])
    if body.doc_idx >= len(docs):
        raise HTTPException(status_code=400, detail="doc_idx out of range")

    doc_name = docs[body.doc_idx]["name"]
    questions, answers, contexts = _extract_qa_triples(
        comparison.get("question_results", []), doc_name
    )

    if not questions:
        raise HTTPException(
            status_code=400, detail="No answerable questions found for this document"
        )

    if len(body.golden_answers) != len(questions):
        raise HTTPException(
            status_code=400,
            detail=f"Provide exactly {len(questions)} golden answers (one per answerable question). "
                   f"Got {len(body.golden_answers)}.",
        )

    dataset = build_ragas_dataset(questions, answers, contexts, references=body.golden_answers)
    metrics = await run_in_threadpool(run_ragas_full, dataset)
    metrics["doc_name"] = doc_name
    metrics["n_questions"] = len(questions)
    metrics["eval_type"] = "full_5_metric"

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {f"ragas_eval.doc_{body.doc_idx}_full": metrics}},
    )
    return metrics


@router.get("/{session_id}/evaluate", status_code=200)
async def get_evaluation(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session = await db.sessions.find_one({"_id": oid, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    evals = session.get("ragas_eval")
    if not evals:
        raise HTTPException(status_code=404, detail="No evaluation run yet")

    return evals
