from pathlib import Path
from typing import List
from pypdf import PdfReader
from app.core.types import Document

def load_local_documents(raw_dir: Path) -> List[Document]:
    docs: List[Document] = []

    for pdf in raw_dir.glob("*.pdf"):
        reader = PdfReader(str(pdf))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        text=text,
                        metadata={"source": str(pdf), "page": i + 1}
                    )
                )
    return docs
