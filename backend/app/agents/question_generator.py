from __future__ import annotations
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class _State(TypedDict):
    title: str
    user_questions: list[str]
    rephrased: list[str]
    generated: list[str]


# Node 1: rephrase user-submitted questions to be precise and perfectly answerable
_REPHRASE_PROMPT = ChatPromptTemplate.from_template(
    "You are helping a user compare options for: {title}\n\n"
    "The user has written the following questions they want answered for each option.\n"
    "Rephrase each question so it is:\n"
    "  - Specific and unambiguous\n"
    "  - Phrased to extract a concrete, measurable answer from a document\n"
    "  - Free of vague words like 'good', 'better', 'nice'\n\n"
    "Output exactly one rephrased question per input question, one per line.\n"
    "Preserve the original meaning and order. No numbering, no bullets.\n\n"
    "User questions:\n{user_questions}"
)

# Node 2: generate 5 additional questions in the direction of the user's intent
_GENERATE_PROMPT = ChatPromptTemplate.from_template(
    "You are designing an evaluation framework to compare options for: {title}\n\n"
    "The user is already asking these questions:\n{rephrased}\n\n"
    "Generate exactly 5 additional questions that:\n"
    "  - Cover important dimensions the user has NOT already asked about\n"
    "  - Are in the same spirit and direction as the user's questions\n"
    "  - Can be answered directly from a document about one of the options\n"
    "  - Are specific enough to produce a measurable, comparable answer\n\n"
    "Output exactly 5 questions, one per line. No numbering, no bullets.\n"
    "Do NOT repeat or rephrase any question already in the list above."
)

_rephrase_chain = _REPHRASE_PROMPT | _llm
_generate_chain = _GENERATE_PROMPT | _llm


async def _rephrase_node(state: _State) -> dict:
    if not state["user_questions"]:
        return {"rephrased": []}
    result = await _rephrase_chain.ainvoke({
        "title": state["title"],
        "user_questions": "\n".join(state["user_questions"]),
    })
    rephrased = [l.strip() for l in result.content.strip().splitlines() if l.strip()]
    return {"rephrased": rephrased}


async def _generate_node(state: _State) -> dict:
    rephrased_block = "\n".join(state["rephrased"]) if state["rephrased"] else "(none)"
    result = await _generate_chain.ainvoke({
        "title": state["title"],
        "rephrased": rephrased_block,
    })
    generated = [l.strip() for l in result.content.strip().splitlines() if l.strip()]
    return {"generated": generated[:5]}


def _build_graph():
    g = StateGraph(_State)
    g.add_node("rephrase", _rephrase_node)
    g.add_node("generate_extra", _generate_node)
    g.add_edge(START, "rephrase")
    g.add_edge("rephrase", "generate_extra")
    g.add_edge("generate_extra", END)
    return g.compile()


_graph = _build_graph()


async def generate_questions(title: str, user_questions: list[str]) -> list[str]:
    """
    Returns rephrased user questions followed by 5 LLM-generated extras.
    All questions are driven by the comparison title and user intent — not document text.
    """
    result = await _graph.ainvoke({
        "title": title,
        "user_questions": user_questions,
        "rephrased": [],
        "generated": [],
    })
    return result["rephrased"] + result["generated"]
