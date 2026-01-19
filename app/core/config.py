from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")

@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 150

@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass(frozen=True)
class VectorStoreConfig:
    index_name: str = "faiss.index"
    meta_name: str = "meta.jsonl"

@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 256
