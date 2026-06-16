from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter

PARENT_CHARS = 1800
PARENT_OVERLAP = 200
CHILD_CHARS = 400
CHILD_OVERLAP = 60


@dataclass
class Chunk:
    text: str
    parent_text: str
    child_index: int
    parent_index: int


_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHARS,
    chunk_overlap=PARENT_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHARS,
    chunk_overlap=CHILD_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_parent_child_chunks(text: str) -> list[Chunk]:
    parents = _parent_splitter.split_text(text)
    chunks: list[Chunk] = []
    for p_idx, parent_text in enumerate(parents):
        children = _child_splitter.split_text(parent_text)
        for c_idx, child_text in enumerate(children):
            chunks.append(
                Chunk(
                    text=child_text,
                    parent_text=parent_text,
                    child_index=c_idx,
                    parent_index=p_idx,
                )
            )
    return chunks
