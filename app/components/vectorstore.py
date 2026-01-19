from pathlib import Path
from typing import List, Dict, Any
import json
import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta: List[Dict[str, Any]] = []

    def build(self, vectors: np.ndarray, texts: List[str], metas: List[Dict[str, Any]]):
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index = index
        self.meta = [{"text": t, "meta": m} for t, m in zip(texts, metas)]

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]

    def search(self, query: np.ndarray, top_k: int):
        scores, idxs = self.index.search(query.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                row = self.meta[idx]
                results.append((score, row))
        return results
