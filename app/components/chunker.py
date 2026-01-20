from __future__ import annotations

from typing import Iterator
from app.core.types import Document

def iter_chunks(doc: Document, chunk_size: int, chunk_overlap: int) -> Iterator[Document]:
    text = doc.text
    n = len(text)
    if n == 0:
        return

    start = 0
    prev_start = -1

    while start < n:
        end = min(n, start + chunk_size)
        chunk_text = text[start:end].strip()

        if chunk_text:
            meta = dict(doc.metadata)
            meta["start"] = start
            meta["end"] = end
            yield Document(text=chunk_text, metadata=meta)

        # Wenn wir das Ende erreicht haben: fertig
        if end >= n:
            break

        # nächsten Start berechnen
        next_start = end - chunk_overlap
        if next_start <= start:
            # Sicherheitsbremse gegen Endlosschleifen
            next_start = end

        prev_start = start
        start = next_start
