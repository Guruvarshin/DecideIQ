from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from app.core.database import get_database
from app.core.dependencies import get_current_user
from app.agents.question_generator import generate_questions

router = APIRouter(prefix="/sessions", tags=["questions"])


class _UserQuestionsBody(BaseModel):
    questions: list[str]


@router.post("/{session_id}/questions", status_code=200)
async def add_user_questions(
    session_id: str,
    body: _UserQuestionsBody,
    current_user: dict = Depends(get_current_user),
):
    """User submits their own questions. Stored separately; used as seed for generation."""
    db = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session = await db.sessions.find_one({"_id": oid, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    cleaned = [q.strip() for q in body.questions if q.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Provide at least one non-empty question")

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {"user_questions": cleaned}},
    )
    return {"user_questions": cleaned, "count": len(cleaned)}


@router.post("/{session_id}/questions/generate", status_code=200)
async def generate_session_questions(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generates questions from the session title + user-submitted questions.
    Does NOT read document text — questions are driven by the comparison intent.
    Returns: rephrased user questions + 5 LLM-generated questions.
    """
    db = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session = await db.sessions.find_one({"_id": oid, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    title = session.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Set a session title before generating questions")

    user_questions = session.get("user_questions", [])

    questions = await generate_questions(title=title, user_questions=user_questions)

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {"generated_questions": questions}},
    )
    return {"questions": questions, "count": len(questions)}


@router.get("/{session_id}/questions", status_code=200)
async def get_session_questions(
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

    return {
        "user_questions": session.get("user_questions", []),
        "generated_questions": session.get("generated_questions", []),
    }
