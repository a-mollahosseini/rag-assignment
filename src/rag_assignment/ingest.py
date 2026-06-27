"""PDF ingestion and Chroma indexing."""

from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_CONFIG, RagConfig
from .deps import missing_dependency_error


def build_embeddings(config: RagConfig = DEFAULT_CONFIG):
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise missing_dependency_error("langchain-ollama") from exc

    return OllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.ollama_base_url,
    )


def load_and_split_documents(config: RagConfig = DEFAULT_CONFIG):
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise missing_dependency_error(
            "langchain-community and langchain-text-splitters"
        ) from exc

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"PDF data directory not found: {data_dir}")

    loader = PyPDFDirectoryLoader(str(data_dir), glob="**/*.pdf")
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF pages were loaded from {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        if source:
            chunk.metadata["source"] = Path(source).name
        chunk.metadata["chunk_id"] = index

    return chunks


def build_vector_store(config: RagConfig = DEFAULT_CONFIG, reset: bool = False):
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise missing_dependency_error("langchain-chroma") from exc

    if reset and Path(config.chroma_dir).exists():
        import shutil

        shutil.rmtree(config.chroma_dir)

    chunks = load_and_split_documents(config)
    embeddings = build_embeddings(config)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.chroma_dir),
        collection_name=config.collection_name,
    )
    return vector_store, chunks


def load_vector_store(config: RagConfig = DEFAULT_CONFIG):
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise missing_dependency_error("langchain-chroma") from exc

    if not Path(config.chroma_dir).exists():
        raise FileNotFoundError(
            f"Chroma database not found at {config.chroma_dir}. Run ingestion first."
        )

    return Chroma(
        embedding_function=build_embeddings(config),
        persist_directory=str(config.chroma_dir),
        collection_name=config.collection_name,
    )


def ingest(config: RagConfig = DEFAULT_CONFIG, reset: bool = False) -> int:
    _, chunks = build_vector_store(config=config, reset=reset)
    return len(chunks)
