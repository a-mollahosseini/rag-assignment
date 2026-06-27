"""Command-line interface for the RAG assignment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, RagConfig


def config_from_args(args: argparse.Namespace) -> RagConfig:
    config = DEFAULT_CONFIG
    overrides = {}
    for field_name in [
        "data_dir",
        "questions_path",
        "chroma_dir",
        "outputs_dir",
        "output_workbook",
    ]:
        value = getattr(args, field_name, None)
        if value:
            overrides[field_name] = Path(value)

    for field_name in [
        "embedding_model",
        "generation_model",
        "collection_name",
        "ollama_base_url",
        "chunk_size",
        "chunk_overlap",
        "top_k",
    ]:
        value = getattr(args, field_name, None)
        if value is not None:
            overrides[field_name] = value

    return replace(config, **overrides)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None, help="Directory containing source PDFs.")
    parser.add_argument("--questions-path", default=None, help="Path to questions .xlsx or .csv file.")
    parser.add_argument("--chroma-dir", default=None, help="Persistent Chroma directory.")
    parser.add_argument("--outputs-dir", default=None, help="Output directory.")
    parser.add_argument("--output-workbook", default=None, help="Evaluation workbook path.")
    parser.add_argument("--embedding-model", default=None, help="Ollama embedding model.")
    parser.add_argument("--generation-model", default=None, help="Ollama generation model.")
    parser.add_argument("--collection-name", default=None, help="Chroma collection name.")
    parser.add_argument("--ollama-base-url", default=None, help="Ollama base URL.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Text splitter chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Text splitter chunk overlap.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG assignment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the Chroma vector index.")
    add_common_args(ingest_parser)
    ingest_parser.add_argument("--reset", action="store_true", help="Delete existing Chroma DB first.")

    ask_parser = subparsers.add_parser("ask", help="Ask one question against the indexed PDFs.")
    add_common_args(ask_parser)
    ask_parser.add_argument("question", help="Question to answer.")

    eval_parser = subparsers.add_parser("evaluate", help="Answer all questions and export Excel results.")
    add_common_args(eval_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.command == "ingest":
        from .ingest import ingest

        count = ingest(config=config, reset=args.reset)
        print(f"Indexed {count} chunks into {config.chroma_dir}")
        return 0

    if args.command == "ask":
        from .rag import answer_question

        result = answer_question(args.question, config=config)
        print(result.answer)
        if result.citations:
            print(f"\nCitations: {result.citation_text}")
        return 0

    if args.command == "evaluate":
        from .evaluate import evaluate

        output_path = evaluate(config=config)
        print(f"Wrote evaluation workbook to {output_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
