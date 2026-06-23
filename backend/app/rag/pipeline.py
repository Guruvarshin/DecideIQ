from __future__ import annotations
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from app.rag.multi_query import generate_sub_queries
from app.rag.hyde import hyde_embedding
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.compressor import compress
from app.rag.grounding import grounding_score, is_grounded
from app.rag.crag import web_search_verified


class _State(TypedDict):
    # ── inputs: set at invocation, never mutated by nodes
    session_id: str
    doc_idx: int
    raw_text: str
    question: str
    will_use_rag: bool
    top_k: int
    # ── intermediate: populated as the graph runs
    sub_queries: list[str]
    hyde_emb: list[float]
    unique_parents: list[str]
    reranked: list[str]
    compressed: list[str]
    g_score: float
    skip_grounding: bool     # True when compressor found content → grounding check unnecessary
    # ── outputs: written by the terminal node on each path
    contexts: list[str]
    source: str
    grounding_score: float


# ── nodes ────────────────────────────────────────────────────────────────────

async def _full_context_node(state: _State) -> dict:
    return {
        "contexts": [state["raw_text"]],
        "source": "full_context",
        "grounding_score": 1.0,
    }


async def _generate_node(state: _State) -> dict:
    sub_queries, hyde_emb = await asyncio.gather(
        generate_sub_queries(state["question"]),
        hyde_embedding(state["question"]),
    )
    return {"sub_queries": sub_queries, "hyde_emb": hyde_emb}


async def _retrieve_node(state: _State) -> dict:
    word_count = len(state["raw_text"].split())
    per_query_k = 5 if word_count <= 3000 else 8

    tasks = [
        retrieve(state["session_id"], state["doc_idx"], state["raw_text"], q, top_k=per_query_k)
        for q in [state["question"]] + state["sub_queries"]
    ]
    tasks.append(
        retrieve(
            state["session_id"], state["doc_idx"], state["raw_text"],
            state["question"], top_k=per_query_k, query_embedding=state["hyde_emb"],
        )
    )
    all_results = await asyncio.gather(*tasks)

    seen: set[str] = set()
    unique_parents: list[str] = []
    for result_list in all_results:
        for p in result_list:
            if p not in seen:
                seen.add(p)
                unique_parents.append(p)

    return {"unique_parents": unique_parents[:20]}


def _rerank_node(state: _State) -> dict:
    return {"reranked": rerank(state["question"], state["unique_parents"], top_k=state["top_k"])}


async def _compress_node(state: _State) -> dict:
    compressed = await compress(state["question"], state["reranked"])
    if compressed:
        # Compressor extracted real sentences directly from passages — guaranteed grounded
        # by construction. No need to run cosine grounding check.
        return {
            "compressed": compressed,
            "contexts": compressed,
            "source": "rag",
            "grounding_score": 1.0,
            "skip_grounding": True,
        }
    else:
        # Compressor found nothing relevant in any passage — fall back to raw reranked
        # passages and let the grounding check decide whether to use them or go to web.
        return {
            "compressed": state["reranked"],
            "skip_grounding": False,
        }


async def _grounding_check_node(state: _State) -> dict:
    score = await grounding_score(state["question"], state["compressed"])
    update: dict = {"g_score": score}
    if is_grounded(score):
        update.update({
            "contexts": state["compressed"],
            "source": "rag",
            "grounding_score": score,
        })
    return update


async def _web_search_node(state: _State) -> dict:
    # web_search_verified runs two checks:
    # 1. Cosine grounding (fast) — catches off-topic results
    # 2. LLM relevance check (meaningful) — catches generic results for document-specific
    #    questions e.g. "What is the best company?" → Tavily returns generic comparison guides
    contexts, source = await web_search_verified(state["question"])
    return {
        "contexts": contexts,
        "source": source,
        "grounding_score": state["g_score"],
    }


# ── routing functions ─────────────────────────────────────────────────────────

def _route_rag_mode(state: _State) -> str:
    return "generate" if state["will_use_rag"] else "full_context"


def _route_after_compress(state: _State) -> str:
    # If compressor extracted content, contexts are already set — skip grounding check.
    # If compressor found nothing, run grounding check on raw reranked passages.
    return "done" if state["skip_grounding"] else "grounding_check"


def _route_grounding(state: _State) -> str:
    return "done" if is_grounded(state["g_score"]) else "web_search"


# ── graph construction ────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(_State)

    g.add_node("full_context", _full_context_node)
    g.add_node("generate", _generate_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("rerank", _rerank_node)
    g.add_node("compress", _compress_node)
    g.add_node("grounding_check", _grounding_check_node)
    g.add_node("web_search", _web_search_node)

    g.add_conditional_edges(
        START, _route_rag_mode,
        {"full_context": "full_context", "generate": "generate"},
    )
    g.add_edge("full_context", END)
    g.add_edge("generate", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "compress")
    g.add_conditional_edges(
        "compress", _route_after_compress,
        {"done": END, "grounding_check": "grounding_check"},
    )
    g.add_conditional_edges(
        "grounding_check", _route_grounding,
        {"done": END, "web_search": "web_search"},
    )
    g.add_edge("web_search", END)

    return g.compile()


_graph = _build_graph()


# ── public API ────────────────────────────────────────────────────────────────

async def run_pipeline(
    session_id: str,
    doc_idx: int,
    raw_text: str,
    question: str,
    will_use_rag: bool,
    top_k: int = 5,
) -> dict:
    result = await _graph.ainvoke({
        "session_id": session_id,
        "doc_idx": doc_idx,
        "raw_text": raw_text,
        "question": question,
        "will_use_rag": will_use_rag,
        "top_k": top_k,
        "sub_queries": [],
        "hyde_emb": [],
        "unique_parents": [],
        "reranked": [],
        "compressed": [],
        "g_score": 0.0,
        "skip_grounding": False,
        "contexts": [],
        "source": "",
        "grounding_score": 0.0,
    })
    return {
        "contexts": result["contexts"],
        "source": result["source"],
        "grounding_score": result["grounding_score"],
    }
