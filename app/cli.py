from app.core.config import Paths, EmbeddingConfig, VectorStoreConfig, LLMConfig
from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.components.llm import LocalLLM
from app.services.rag_service import RAGService

def main():
    paths = Paths()

    embedder = SentenceTransformerEmbedder(EmbeddingConfig().model_name)
    store = FaissVectorStore(
        paths.index_dir / VectorStoreConfig().index_name,
        paths.index_dir / VectorStoreConfig().meta_name,
    )
    store.load()

    llm = LocalLLM(LLMConfig().model_name, LLMConfig().max_new_tokens)
    rag = RAGService(embedder, store, llm)

    print("📚 FDM Chatbot (CLI)")
    print("exit zum Beenden\n")

    while True:
        q = input("Frage> ").strip()
        if q.lower() == "exit":
            break

        answer, sources = rag.answer(q, top_k=5)
        print("\nAntwort:\n", answer, "\n")

if __name__ == "__main__":
    main()
