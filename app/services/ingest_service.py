from __future__ import annotations

from typing import Iterator, List
from app.core.types import Document
from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore

class IngestService:
    def __init__(self, embedder: SentenceTransformerEmbedder, store: FaissVectorStore, batch_size: int = 64):
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    def run(self, chunks_iter: Iterator[Document]) -> int:
        buffer_texts: List[str] = []
        buffer_metas: List[dict] = []
        total = 0

        for chunk in chunks_iter:
            buffer_texts.append(chunk.text)
            buffer_metas.append(chunk.metadata)

            if len(buffer_texts) >= self.batch_size:
                vecs = self.embedder.embed(buffer_texts)
                self.store.add_batch(vecs, buffer_texts, buffer_metas)
                total += len(buffer_texts)
                buffer_texts, buffer_metas = [], []

        # Rest
        if buffer_texts:
            vecs = self.embedder.embed(buffer_texts)
            self.store.add_batch(vecs, buffer_texts, buffer_metas)
            total += len(buffer_texts)

        self.store.save()
        return total
