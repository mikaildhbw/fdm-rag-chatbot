from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.components.llm import LocalLLM

class RAGService:
    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def answer(self, question: str, top_k: int):
        q_vec = self.embedder.embed([question])[0]
        results = self.store.search(q_vec, top_k)

        context = "\n\n".join(r[1]["text"] for r in results)

        prompt = (
            "Beantworte die Frage ausschließlich anhand des Kontexts.\n\n"
            f"Kontext:\n{context}\n\n"
            f"Frage: {question}\nAntwort:"
        )

        out = self.llm.generate(prompt)
        return out.split("Antwort:", 1)[-1].strip(), results
