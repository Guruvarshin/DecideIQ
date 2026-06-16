from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)

_PROMPT = ChatPromptTemplate.from_template(
    "You are a senior decision analyst delivering a final verdict.\n\n"
    "The user is comparing: {doc_names}\n\n"
    "WINNER: {winner_name} ({winner_pct}%)\n\n"
    "Score breakdown:\n{score_summary}\n\n"
    "Question-by-question analysis:\n{qa_breakdown}\n\n"
    "Write a verdict in 3-5 paragraphs:\n"
    "1. Open with a direct winner declaration and the score margin\n"
    "2. Explain the 2-3 key reasons the winner won — cite specific scores and exact figures\n"
    "3. Acknowledge the runner-up's genuine strengths (dimensions where it scored higher)\n"
    "4. Give a nuanced recommendation — when would someone rationally choose the runner-up?\n"
    "5. One final sentence starting with 'Bottom line:'\n\n"
    "Be specific. Use exact figures from the breakdown. No generic filler.\n"
    "IMPORTANT: Never use em dashes (— or --). Use a comma or rewrite the sentence instead."
)

_chain = _PROMPT | _llm


def _format_scores(doc_summaries: list[dict]) -> str:
    return "\n".join(
        f"  {ds['doc_name']}: {ds['raw_score']} pts ({ds['percentage']}%)"
        for ds in doc_summaries
    )


def _format_qa(question_results: list[dict]) -> str:
    lines = []
    for qr in question_results:
        lines.append(f"Q: {qr['question']}")
        for pd in qr["per_doc"]:
            lines.append(f"  {pd['doc_name']} [{pd['score']}/10]: {pd['answer'][:120]}")
    return "\n".join(lines)


async def generate_verdict(comparison_result: dict) -> str:
    summaries = comparison_result["doc_summaries"]
    winner_name = comparison_result["winner_name"]
    winner_pct = next(
        ds["percentage"] for ds in summaries if ds["doc_name"] == winner_name
    )
    result = await _chain.ainvoke({
        "doc_names": " vs ".join(ds["doc_name"] for ds in summaries),
        "winner_name": winner_name,
        "winner_pct": winner_pct,
        "score_summary": _format_scores(summaries),
        "qa_breakdown": _format_qa(comparison_result["question_results"]),
    })
    return result.content.strip()
