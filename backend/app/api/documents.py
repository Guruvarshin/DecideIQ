from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from typing import Optional
from bson import ObjectId
from datetime import datetime, timezone
from app.core.database import get_database
from app.core.dependencies import get_current_user
from app.ingestion.document_processor import process_upload, process_text_paste
from app.rag.chunker import build_parent_child_chunks
from app.rag.embedder import embed_texts
from app.rag.vector_store import add_chunks, delete_doc_collection

router = APIRouter(prefix="/sessions", tags=["documents"])


async def _get_pending_session(session_id: str, user_id: ObjectId, db):
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    session = await db.sessions.find_one({"_id": oid, "user_id": user_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "pending":
        raise HTTPException(status_code=400, detail="Session is not accepting documents")
    return session


@router.post("/{session_id}/documents", status_code=201)
async def add_document(
    session_id: str,
    file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None),
    document_name: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    db = get_database()
    session = await _get_pending_session(session_id, current_user["_id"], db)

    if file and file.filename:
        content = await file.read()
        try:
            doc = process_upload(content, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if document_name:
            doc["name"] = document_name
    elif text_content:
        doc = process_text_paste(text_content, document_name or "Pasted Document")
    else:
        raise HTTPException(status_code=400, detail="Provide a file or text_content")

    doc["uploaded_at"] = datetime.now(timezone.utc)
    idx = len(session.get("documents", []))

    if doc.get("will_use_rag"):
        chunks = build_parent_child_chunks(doc["raw_text"])
        embeddings = await embed_texts([c.text for c in chunks])
        add_chunks(session_id, idx, chunks, embeddings)
        doc["chunk_count"] = len(chunks)

    await db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$push": {"documents": doc}},
    )

    return {
        "idx": idx,
        "name": doc["name"],
        "source_type": doc["source_type"],
        "word_count": doc["word_count"],
        "will_use_rag": doc["will_use_rag"],
        "chunk_count": doc.get("chunk_count"),
    }


@router.delete("/{session_id}/documents/{doc_idx}")
async def remove_document(
    session_id: str,
    doc_idx: int,
    current_user: dict = Depends(get_current_user),
):
    db = get_database()
    session = await _get_pending_session(session_id, current_user["_id"], db)
    docs = session.get("documents", [])
    if doc_idx < 0 or doc_idx >= len(docs):
        raise HTTPException(status_code=404, detail="Document not found")
    docs.pop(doc_idx)
    delete_doc_collection(session_id, doc_idx)
    await db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"documents": docs}},
    )
    return {"message": "Document removed"}
