from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime, timezone
from app.core.database import get_database
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/sessions", tags=["sessions"])


class _CreateSessionBody(BaseModel):
    title: str = ""


@router.post("", status_code=201)
async def create_session(
    body: _CreateSessionBody,
    current_user: dict = Depends(get_current_user),
):
    db = get_database()
    result = await db.sessions.insert_one({
        "user_id": current_user["_id"],
        "status": "pending",
        "title": body.title.strip(),
        "documents": [],
        "user_questions": [],
        "generated_questions": [],
        "created_at": datetime.now(timezone.utc),
    })
    return {"session_id": str(result.inserted_id), "title": body.title.strip()}


@router.get("")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    db = get_database()
    cursor = db.sessions.find(
        {"user_id": current_user["_id"]},
        {"documents.raw_text": 0},
    ).sort("created_at", -1)
    sessions = []
    async for s in cursor:
        sessions.append({
            "session_id": str(s["_id"]),
            "title": s.get("title", ""),
            "status": s["status"],
            "document_count": len(s.get("documents", [])),
            "created_at": s["created_at"],
        })
    return sessions


@router.get("/{session_id}")
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    session = await db.sessions.find_one(
        {"_id": oid, "user_id": current_user["_id"]},
        {"documents.raw_text": 0},
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": str(session["_id"]),
        "title": session.get("title", ""),
        "status": session["status"],
        "documents": [
            {k: v for k, v in doc.items() if k != "raw_text"}
            for doc in session.get("documents", [])
        ],
        "questions": session.get("questions", []),
        "created_at": session["created_at"],
    }
