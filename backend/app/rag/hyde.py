from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.rag.embedder import embed_query

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

_PROMPT = ChatPromptTemplate.from_template(
    "Write a short paragraph (3-5 sentences) that would answer the following question "
    "if it appeared verbatim inside a document. Write as the document, not as an assistant.\n\n"
    "Question: {question}"
)

_chain = _PROMPT | _llm


async def hyde_embedding(question: str) -> list[float]:
    result = await _chain.ainvoke({"question": question})
    hypothetical_doc = result.content.strip()
    return await embed_query(hypothetical_doc)
