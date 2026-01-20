from __future__ import annotations

from pathlib import Path
from typing import Iterator
from pypdf import PdfReader
from app.core.types import Document

def iter_local_documents(raw_dir: Path) -> Iterator[Document]:
    for pdf in raw_dir.glob("*.pdf"):
        reader = PdfReader(str(pdf))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                yield Document(text=text, metadata={"source": str(pdf), "page": i + 1})
