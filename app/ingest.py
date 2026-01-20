from __future__ import annotations

from app.core.config import Paths, ChunkingConfig, EmbeddingConfig, VectorStoreConfig
from app.components.loaders import iter_local_documents
from app.components.chunker import iter_chunks
from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.services.ingest_service import IngestService

def main():
    paths = Paths()
    chunk_cfg = ChunkingConfig()
    emb_cfg = EmbeddingConfig()
    vs_cfg = VectorStoreConfig()

    embedder = SentenceTransformerEmbedder(emb_cfg.model_name)
    store = FaissVectorStore(paths.index_dir / vs_cfg.index_name, paths.index_dir / vs_cfg.meta_name)

    def all_chunks():
        for doc in iter_local_documents(paths.raw_dir):
            yield from iter_chunks(doc, chunk_cfg.chunk_size, chunk_cfg.chunk_overlap)

    total = IngestService(embedder, store, batch_size=64).run(all_chunks())
    print(f"✅ Ingest fertig: {total} Chunks. Index gespeichert in {paths.index_dir}")

if __name__ == "__main__":
    main()
