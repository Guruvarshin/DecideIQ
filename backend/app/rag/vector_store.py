from __future__ import annotations
import logging
import time
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from app.rag.chunker import Chunk

_pc: Pinecone | None = None
_index = None

_EMPTY_QUERY_RESULT = {"ids": [[]]}

logger = logging.getLogger(__name__)


def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        name = settings.pinecone_index_name

        existing = [i.name for i in _pc.list_indexes()]
        if name not in existing:
            logger.info("Creating Pinecone index '%s'...", name)
            _pc.create_index(
                name=name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not _pc.describe_index(name).status["ready"]:
                time.sleep(1)
            logger.info("Pinecone index '%s' ready.", name)

        _index = _pc.Index(name)
    return _index


def _namespace(session_id: str, doc_idx: int) -> str:
    return f"s{session_id}_d{doc_idx}"


def add_chunks(
    session_id: str,
    doc_idx: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    index = _get_index()
    ns = _namespace(session_id, doc_idx)
    vectors = [
        {
            "id": f"{doc_idx}_{i}",
            "values": embeddings[i],
            "metadata": {
                "parent_text":  chunks[i].parent_text,
                "parent_index": chunks[i].parent_index,
            },
        }
        for i in range(len(chunks))
    ]
    # Upsert in batches of 100 (Pinecone 2MB/request limit)
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100], namespace=ns)


def query_dense(
    session_id: str,
    doc_idx: int,
    query_embedding: list[float],
    n_results: int = 20,
) -> dict:
    try:
        result = _get_index().query(
            vector=query_embedding,
            top_k=n_results,
            namespace=_namespace(session_id, doc_idx),
            include_metadata=False,
        )
        return {"ids": [[m.id for m in result.matches]]}
    except Exception:
        logger.warning(
            "Pinecone namespace %s not found — dense retrieval skipped, "
            "falling back to BM25 only.",
            _namespace(session_id, doc_idx),
        )
        return _EMPTY_QUERY_RESULT


def delete_doc_collection(session_id: str, doc_idx: int) -> None:
    try:
        _get_index().delete(
            delete_all=True,
            namespace=_namespace(session_id, doc_idx),
        )
    except Exception:
        pass
