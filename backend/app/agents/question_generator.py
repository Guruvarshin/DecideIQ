from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

_REPHRASE_PROMPT = ChatPromptTemplate.from_template(
    "You are helping a user compare options for: {title}\n\n"
    "Rephrase each user question so it is specific, unambiguous, and extractable from a document.\n"
    "Output exactly one rephrased question per input question, one per line.\n"
    "No numbering, no bullets.\n\n"
    "User questions:\n{user_questions}"
)

_PER_DOC_PROMPT = ChatPromptTemplate.from_template(
    "You are analyzing a document to help a user compare options for: {title}\n\n"
    "Document excerpt:\n{text}\n\n"
    "Generate exactly 5 questions whose answers can be found in this document and would help "
    "someone decide whether this option is the right choice.\n"
    "Requirements:\n"
    "- Each question must be answerable with a concrete fact or figure from the document\n"
    "- Cover different dimensions (cost, terms, benefits, risks, eligibility, etc.)\n"
    "- Phrase each question so it can also be asked to other similar options for comparison\n"
    "- No vague words like 'good', 'better', 'nice'\n\n"
    "Output exactly 5 questions, one per line. No numbering, no bullets."
)

_DEDUP_PROMPT = ChatPromptTemplate.from_template(
    "You are merging question lists for a comparison of: {title}\n\n"
    "Here are questions generated from {n} documents:\n{questions}\n\n"
    "Remove questions that are duplicates or near-duplicates of each other, keeping the clearest phrasing.\n"
    "Return the deduplicated list, one question per line. No numbering, no bullets.\n"
    "Preserve all genuinely distinct questions."
)

_rephrase_chain = _REPHRASE_PROMPT | _llm
_per_doc_chain  = _PER_DOC_PROMPT | _llm
_dedup_chain    = _DEDUP_PROMPT | _llm


async def generate_questions(
    title: str,
    user_questions: list[str],
    doc_texts: list[str],
) -> list[str]:
    """
    Returns rephrased user questions + deduplicated questions generated from each document's content.
    With n docs, generates n*5 questions then deduplicates to ~8-15 unique ones.
    """
    import asyncio

    # Step 1: rephrase user questions
    rephrased: list[str] = []
    if user_questions:
        r = await _rephrase_chain.ainvoke({
            "title": title,
            "user_questions": "\n".join(user_questions),
        })
        rephrased = [l.strip() for l in r.content.strip().splitlines() if l.strip()]

    # Step 2: generate 5 questions per document concurrently
    tasks = [
        _per_doc_chain.ainvoke({
            "title": title,
            "text": text[:3000],  # cap to avoid token overflow
        })
        for text in doc_texts
    ]
    results = await asyncio.gather(*tasks)

    per_doc_questions: list[str] = []
    for res in results:
        qs = [l.strip() for l in res.content.strip().splitlines() if l.strip()]
        per_doc_questions.extend(qs[:5])

    # Step 3: deduplicate across all per-doc questions
    if per_doc_questions:
        combined = "\n".join(per_doc_questions)
        d = await _dedup_chain.ainvoke({
            "title": title,
            "n": len(doc_texts),
            "questions": combined,
        })
        deduped = [l.strip() for l in d.content.strip().splitlines() if l.strip()]
    else:
        deduped = []

    # Rephrased user questions come first, then doc-derived questions
    # Skip deduped questions that overlap with rephrased ones
    rephrased_lower = {q.lower() for q in rephrased}
    unique_doc_qs = [q for q in deduped if q.lower() not in rephrased_lower]

    return rephrased + unique_doc_qs
