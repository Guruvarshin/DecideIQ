from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_PROMPT = ChatPromptTemplate.from_template(
    "You are helping retrieve information from a document.\n"
    "Generate 3 different sub-questions that together cover all aspects of the original question.\n"
    "Output exactly 3 questions, one per line, no numbering, no bullets.\n\n"
    "Original question: {question}"
)

_chain = _PROMPT | _llm


async def generate_sub_queries(question: str) -> list[str]:
    result = await _chain.ainvoke({"question": question})
    lines = [l.strip() for l in result.content.strip().splitlines() if l.strip()]
    return lines[:3]
