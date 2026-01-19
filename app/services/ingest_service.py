from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.core.types import Document
from typing import List

class IngestService:
    def __init__(self, embedder: SentenceTransformerEmbedder, store: FaissVectorStore):
        self.embedder = embedder
        self.store = store

    def run(self, docs: List[Document]):
        texts = [d.text for d in docs]
        metas = [d.metadata for d in docs]
        vectors = self.embedder.embed(texts)
        self.store.build(vectors, texts, metas)
