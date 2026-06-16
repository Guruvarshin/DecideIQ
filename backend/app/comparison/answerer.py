from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question below using ONLY the provided context.\n\n"
    "Rules:\n"
    "- Copy exact figures verbatim: currency amounts, percentages, days, periods, ratios "
    "(e.g. ₹18,00,000 per annum not '18 lakhs', 12% not 'twelve percent', 26 weeks not '6 months')\n"
    "- Include ALL relevant numbers mentioned in the context for this question — do not pick just one\n"
    "- If the question asks about multiple items (e.g. health + life insurance), cover each one\n"
    "- Be concise but complete — one to three sentences\n"
    "- If the context does not contain the answer, respond with exactly: Not found in document.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)

_chain = _PROMPT | _llm


async def answer_question(question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    result = await _chain.ainvoke({"question": question, "context": context_text})
    return result.content.strip()
