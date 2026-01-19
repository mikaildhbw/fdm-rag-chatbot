from typing import List
from app.core.types import Document

def chunk_documents(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:

    chunks: List[Document] = []

    for doc in docs:
        text = doc.text
        start = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()

            if chunk_text:
                meta = dict(doc.metadata)
                meta["start"] = start
                meta["end"] = end
                chunks.append(Document(text=chunk_text, metadata=meta))

            start = end - chunk_overlap
            if start < 0:
                start = 0

    return chunks
