from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISS Vector Store with:
    - incremental add_batch during ingest
    - persisted FAISS index on disk
    - metadata stored line-by-line in JSONL (meta.jsonl)
    """

    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path

        self.index: faiss.Index | None = None
        self.meta: List[Dict[str, Any]] = []

    def init(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized embeddings (IP)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # overwrite meta file
        with self.meta_path.open("w", encoding="utf-8") as f:
            pass

        self.meta = []

    def add_batch(self, vectors: np.ndarray, texts: List[str], metas: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array of shape (n, dim)")

        if self.index is None:
            self.init(vectors.shape[1])

        # add to index
        self.index.add(vectors.astype("float32"))

        # append metadata
        with self.meta_path.open("a", encoding="utf-8") as f:
            for t, m in zip(texts, metas):
                row = {"text": t, "meta": m}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized. Nothing to save.")
        faiss.write_index(self.index, str(self.index_path))

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}. Run ingest first.")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}. Run ingest first.")

        self.index = faiss.read_index(str(self.index_path))

        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.meta.append(json.loads(line))

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Returns a list of tuples: (score, row) where row={"text":..., "meta":...}
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores, idxs = self.index.search(query_vec.astype("float32"), top_k)

        results: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            # safety check (should always be in range if ingest wrote correctly)
            if idx >= len(self.meta):
                continue
            results.append((float(score), self.meta[idx]))
        return results
