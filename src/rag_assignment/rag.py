"""Retrieval-augmented answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_CONFIG, RagConfig
from .deps import missing_dependency_error
from .ingest import load_vector_store


@dataclass(frozen=True)
class Citation:
    source: str
    page: int | None
    chunk_id: int | None

    @property
    def label(self) -> str:
        page_text = f", page {self.page + 1}" if self.page is not None else ""
        chunk_text = f", chunk {self.chunk_id}" if self.chunk_id is not None else ""
        return f"{self.source}{page_text}{chunk_text}"


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: list[Citation]
    contexts: list[str]
    top_source: str | None
    top_page: int | None

    @property
    def citation_text(self) -> str:
        return "; ".join(c.label for c in self.citations)


def _make_llm(config: RagConfig = DEFAULT_CONFIG):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise missing_dependency_error("langchain-ollama") from exc

    return ChatOllama(
        model=config.generation_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )


def _citation_from_metadata(metadata: dict) -> Citation:
    source = metadata.get("source") or metadata.get("file_path") or "unknown"
    source = Path(str(source)).name
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")
    return Citation(
        source=source,
        page=int(page) if page is not None else None,
        chunk_id=int(chunk_id) if chunk_id is not None else None,
    )


def _format_contexts(docs: Iterable) -> tuple[str, list[Citation], list[str]]:
    citations: list[Citation] = []
    context_texts: list[str] = []
    blocks: list[str] = []

    for i, doc in enumerate(docs, start=1):
        citation = _citation_from_metadata(doc.metadata)
        citations.append(citation)
        text = " ".join((doc.page_content or "").split())
        context_texts.append(text)
        blocks.append(f"[{i}] Source: {citation.label}\n{text}")

    return "\n\n".join(blocks), citations, context_texts


def build_prompt(question: str, context_block: str) -> str:
    return f"""You are answering questions about clinical-trial protocol PDFs.
Use only the retrieved context below. If the answer is not supported by the
context, say that the provided context does not contain enough evidence.

Write a concise answer. Include citations using bracket numbers like [1] or [2].

Retrieved context:
{context_block}

Question:
{question}

Answer:"""


def answer_question(question: str, config: RagConfig = DEFAULT_CONFIG) -> RagAnswer:
    vector_store = load_vector_store(config)
    retriever = vector_store.as_retriever(search_kwargs={"k": config.top_k})
    docs = retriever.invoke(question)

    context_block, citations, contexts = _format_contexts(docs)
    if not docs:
        return RagAnswer(
            question=question,
            answer="No relevant context was retrieved.",
            citations=[],
            contexts=[],
            top_source=None,
            top_page=None,
        )

    llm = _make_llm(config)
    response = llm.invoke(build_prompt(question, context_block))
    answer = getattr(response, "content", str(response)).strip()
    top_citation = citations[0]
    return RagAnswer(
        question=question,
        answer=answer,
        citations=citations,
        contexts=contexts,
        top_source=top_citation.source,
        top_page=top_citation.page + 1 if top_citation.page is not None else None,
    )
