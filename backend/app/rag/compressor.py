from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_PROMPT = ChatPromptTemplate.from_template(
    "Extract only the sentences from the passage below that directly help answer the question.\n"
    "Return only the extracted sentences, nothing else.\n"
    "If nothing in the passage is relevant, return an empty string.\n\n"
    "Question: {question}\n"
    "Passage: {passage}"
)

_chain = _PROMPT | _llm


async def compress(question: str, passages: list[str]) -> list[str]:
    compressed: list[str] = []
    for passage in passages:
        result = await _chain.ainvoke({"question": question, "passage": passage})
        text = result.content.strip()
        if text:
            compressed.append(text)
    return compressed
