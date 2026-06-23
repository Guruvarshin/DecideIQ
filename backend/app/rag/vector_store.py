from __future__ import annotations
import chromadb
from app.core.config import settings
from app.rag.chunker import Chunk

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_path)
    return _client


def _col_name(session_id: str, doc_idx: int) -> str:
    return f"s{session_id}_d{doc_idx}"


def add_chunks(
    session_id: str,
    doc_idx: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    col = _get_client().get_or_create_collection(
        name=_col_name(session_id, doc_idx),
        metadata={"hnsw:space": "cosine"},
    )
    col.add(
        ids=[f"{doc_idx}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"parent_text": c.parent_text, "parent_index": c.parent_index}
            for c in chunks
        ],
    )


_EMPTY_QUERY_RESULT: chromadb.QueryResult = {
    "ids": [[]],
    "embeddings": None,
    "documents": [[]],
    "metadatas": [[]],
    "distances": [[]],
    "uris": None,
    "data": None,
    "included": ["documents", "metadatas", "distances"],
}


def query_dense(
    session_id: str,
    doc_idx: int,
    query_embedding: list[float],
    n_results: int = 20,
) -> chromadb.QueryResult:
    try:
        col = _get_client().get_collection(_col_name(session_id, doc_idx))
    except Exception:
        # Collection missing — container likely restarted and wiped chroma_data.
        # Return empty result so retriever degrades to BM25-only instead of crashing.
        import logging
        logging.getLogger(__name__).warning(
            "ChromaDB collection %s not found — dense retrieval skipped, "
            "falling back to BM25 only. Re-upload documents to restore full retrieval.",
            _col_name(session_id, doc_idx),
        )
        return _EMPTY_QUERY_RESULT
    return col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def delete_doc_collection(session_id: str, doc_idx: int) -> None:
    try:
        _get_client().delete_collection(_col_name(session_id, doc_idx))
    except Exception:
        pass
