from __future__ import annotations

from app.core.config import Paths, EmbeddingConfig, VectorStoreConfig, LLMConfig
from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.components.llm import LocalLLM
from app.services.rag_service import RAGService


def main() -> None:
    paths = Paths()
    emb_cfg = EmbeddingConfig()
    vs_cfg = VectorStoreConfig()
    llm_cfg = LLMConfig()

    print("📚 FDM RAG Chatbot (CLI)")
    print("Lade Index, Embeddings und LLM ...")

    embedder = SentenceTransformerEmbedder(emb_cfg.model_name)

    store = FaissVectorStore(
        index_path=paths.index_dir / vs_cfg.index_name,
        meta_path=paths.index_dir / vs_cfg.meta_name,
    )
    store.load()

    llm = LocalLLM(llm_cfg.model_name, llm_cfg.max_new_tokens)

    rag = RAGService(embedder, store, llm)

    print("✅ Bereit. Tippe deine Frage. Beenden mit: exit\n")

    while True:
        q = input("Frage> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        answer, sources = rag.answer(q, top_k=5)

        print("\nAntwort:\n" + answer + "\n")

        print("Quellen:")
        for score, row in sources:
            meta = row.get("meta", {})
            src = meta.get("source", "-")
            page = meta.get("page", "-")
            print(f"- {src} (Seite {page}) | Score={score:.3f}")

        print("\n" + ("-" * 60) + "\n")


if __name__ == "__main__":
    main()
