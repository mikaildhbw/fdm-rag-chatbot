from app.core.config import Paths, ChunkingConfig, EmbeddingConfig, VectorStoreConfig
from app.components.loaders import load_local_documents
from app.components.chunker import chunk_documents
from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.services.ingest_service import IngestService

def main():
    paths = Paths()
    chunk_cfg = ChunkingConfig()

    docs = load_local_documents(paths.raw_dir)
    chunks = chunk_documents(docs, chunk_cfg.chunk_size, chunk_cfg.chunk_overlap)

    embedder = SentenceTransformerEmbedder(EmbeddingConfig().model_name)
    store = FaissVectorStore(
        paths.index_dir / VectorStoreConfig().index_name,
        paths.index_dir / VectorStoreConfig().meta_name,
    )

    IngestService(embedder, store).run(chunks)
    print(f"✅ Ingest fertig: {len(chunks)} Chunks")

if __name__ == "__main__":
    main()
