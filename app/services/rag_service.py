from __future__ import annotations

from typing import List, Tuple, Dict, Any

from app.components.embeddings import SentenceTransformerEmbedder
from app.components.vectorstore import FaissVectorStore
from app.components.llm import LocalLLM


class RAGService:
    def __init__(self, embedder: SentenceTransformerEmbedder, store: FaissVectorStore, llm: LocalLLM):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def answer(self, question: str, top_k: int = 5) -> Tuple[str, List[Tuple[float, Dict[str, Any]]]]:
        q_vec = self.embedder.embed([question])[0]
        results = self.store.search(q_vec, top_k=top_k)

        context = "\n\n---\n\n".join(row["text"] for _, row in results)

        prompt = (
            "Du bist ein hilfreicher Assistent für Forschungsdatenmanagement (FDM).\n"
            "Beantworte die Frage ausschließlich mit Hilfe des gegebenen Kontexts.\n"
            "Wenn der Kontext nicht reicht, sage klar, dass dir Informationen fehlen.\n\n"
            f"Kontext:\n{context}\n\n"
            f"Frage: {question}\n"
            "Antwort:"
        )

        generated = self.llm.generate(prompt)

        # Many pipelines return prompt+answer; strip to the part after "Antwort:"
        answer = generated.split("Antwort:", 1)[-1].strip()
        return answer, results
