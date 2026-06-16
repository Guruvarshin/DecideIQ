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
    # ── intermediate: populated as the graph runs
    sub_queries: list[str]
    hyde_emb: list[float]
    unique_parents: list[str]
    reranked: list[str]
    compressed: list[str]
    g_score: float          # raw grounding score, read by routing + web_search node
    # ── outputs: written by the terminal node on each path
    contexts: list[str]
    source: str
    grounding_score: float  # exposed in the public return dict


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
    per_query_k = 3 if word_count <= 3000 else (5 if word_count <= 8000 else 8)

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

    return {"unique_parents": unique_parents}


def _rerank_node(state: _State) -> dict:
    return {"reranked": rerank(state["question"], state["unique_parents"], top_k=8)}


async def _compress_node(state: _State) -> dict:
    compressed = await compress(state["question"], state["reranked"])
    return {"compressed": compressed if compressed else state["reranked"]}


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
    contexts, source = await web_search_verified(state["question"])
    return {
        "contexts": contexts,
        "source": source,
        "grounding_score": state["g_score"],
    }


# ── routing functions ─────────────────────────────────────────────────────────

def _route_rag_mode(state: _State) -> str:
    return "generate" if state["will_use_rag"] else "full_context"


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
    g.add_edge("compress", "grounding_check")
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
) -> dict:
    result = await _graph.ainvoke({
        "session_id": session_id,
        "doc_idx": doc_idx,
        "raw_text": raw_text,
        "question": question,
        "will_use_rag": will_use_rag,
        "sub_queries": [],
        "hyde_emb": [],
        "unique_parents": [],
        "reranked": [],
        "compressed": [],
        "g_score": 0.0,
        "contexts": [],
        "source": "",
        "grounding_score": 0.0,
    })
    return {
        "contexts": result["contexts"],
        "source": result["source"],
        "grounding_score": result["grounding_score"],
    }
