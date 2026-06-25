from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from app.core.database import get_database
from app.core.dependencies import get_current_user
from app.comparison.engine import run_comparison
from app.comparison.verdict import generate_verdict
from app.rag.vector_store import delete_doc_collection

router = APIRouter(prefix="/sessions", tags=["comparison"])


@router.post("/{session_id}/compare", status_code=200)
async def compare_session(
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
    if not session.get("generated_questions"):
        raise HTTPException(status_code=400, detail="Generate questions first")
    if len(session.get("documents", [])) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 documents")

    try:
        result = await run_comparison(session_id, session)
        result["verdict"] = await generate_verdict(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["compared_at"] = datetime.now(timezone.utc)

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {"comparison_result": result, "status": "compared"}},
    )

    # Delete Pinecone namespaces now that comparison is done — results are stored
    # in MongoDB so vectors are no longer needed.
    n_docs = len(session.get("documents", []))
    for i in range(n_docs):
        delete_doc_collection(session_id, i)

    return result


@router.get("/{session_id}/compare", status_code=200)
async def get_comparison(
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

    result = session.get("comparison_result")
    if not result:
        raise HTTPException(status_code=404, detail="No comparison run yet")

    return result
