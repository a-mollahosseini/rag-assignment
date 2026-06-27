"""Configuration defaults for the RAG assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    """Return the repository root when running from source or an editable install."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RagConfig:
    root_dir: Path = project_root()
    data_dir: Path = root_dir / "data"
    questions_path: Path = root_dir / "Questions.xlsx"
    chroma_dir: Path = root_dir / "chroma_db"
    outputs_dir: Path = root_dir / "outputs"
    output_workbook: Path = outputs_dir / "rag_answers.xlsx"
    collection_name: str = "rag_assignment_clinical_trials"
    embedding_model: str = "mxbai-embed-large"
    generation_model: str = "mistral"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    ollama_base_url: str = "http://localhost:11434"


DEFAULT_CONFIG = RagConfig()
