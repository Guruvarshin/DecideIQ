from __future__ import annotations
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_NOT_FOUND_PHRASES = (
    "not found in document",
    "not mentioned in available sources",
    "no information",
)

_PROMPT = ChatPromptTemplate.from_template(
    "You are scoring answers for a decision-support system.\n"
    "Question: {question}\n\n"
    "The following answers come from {n} different options being compared:\n"
    "{answers}\n\n"
    "Score each answer from 1 to 10 based on how favorable it is for the decision-maker.\n"
    "Rules:\n"
    "- Higher score = better outcome for someone making this choice\n"
    "- Score relative to each other, not in isolation\n"
    "- 'Not found in document' / 'Not mentioned' answers score 1\n"
    "- Use the full 1-10 range to differentiate clearly\n\n"
    "Output ONLY a JSON array of integers with exactly {n} elements, e.g. [7, 9]\n"
    "One score per answer, in the same order as listed above."
)

_chain = _PROMPT | _llm


def _parse_scores(text: str, n: int) -> list[int]:
    match = re.search(r"\[[\d,\s]+\]", text)
    if match:
        try:
            scores = json.loads(match.group())
            if len(scores) == n and all(isinstance(s, int) for s in scores):
                return [max(1, min(10, s)) for s in scores]
        except (json.JSONDecodeError, TypeError):
            pass
    return [5] * n


def _is_not_found(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in _NOT_FOUND_PHRASES)


async def score_answers(question: str, answers: list[str]) -> list[int]:
    n = len(answers)
    # If every doc failed to find an answer, return neutral 5s — the question
    # doesn't differentiate and shouldn't drag down all scores equally.
    if all(_is_not_found(a) for a in answers):
        return [5] * n
    formatted = "\n".join(
        f"Option {i + 1}: {a}" for i, a in enumerate(answers)
    )
    result = await _chain.ainvoke({"question": question, "n": n, "answers": formatted})
    return _parse_scores(result.content, n)
