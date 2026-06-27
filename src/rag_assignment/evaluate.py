"""Question answering and retrieval evaluation workbook export."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, RagConfig
from .deps import missing_dependency_error
from .rag import RagAnswer, answer_question


QUESTION_COLUMNS = {"question", "questions", "query"}
ANSWER_COLUMNS = {"answer", "answers", "gold_answer", "reference_answer", "expected_answer"}


@dataclass(frozen=True)
class QuestionRow:
    index: int
    question: str
    reference_answer: str | None = None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def exact_match(predicted: str, expected: str | None) -> str:
    if not expected:
        return "N/A"
    return str(_normalize_text(predicted) == _normalize_text(expected))


def hallucination_flag(answer: str, contexts: list[str]) -> str:
    unsupported_markers = [
        "does not contain enough evidence",
        "not provided",
        "not specified",
        "cannot determine",
        "no relevant context",
    ]
    normalized_answer = answer.lower()
    if any(marker in normalized_answer for marker in unsupported_markers):
        return "Needs review"

    answer_terms = {
        token
        for token in re.findall(r"[a-zA-Z0-9]{4,}", normalized_answer)
        if token not in {"that", "this", "with", "from", "were", "will", "study"}
    }
    context_text = " ".join(contexts).lower()
    missing_terms = [term for term in answer_terms if term not in context_text]
    if answer_terms and len(missing_terms) / len(answer_terms) > 0.45:
        return "Needs review"
    return "No obvious hallucination"


def recall_at_1(question: str, top_source: str | None) -> str:
    trial_ids = re.findall(r"NCT\d{8}", question.upper())
    if not trial_ids:
        return "N/A"
    if not top_source:
        return "False"
    return str(any(trial_id in top_source.upper() for trial_id in trial_ids))


def read_questions(path: Path) -> list[QuestionRow]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise missing_dependency_error("pandas and openpyxl") from exc

    if not path.exists():
        raise FileNotFoundError(f"Questions workbook not found: {path}")

    df = pd.read_excel(path)
    if df.empty:
        raise ValueError(f"No questions found in {path}")

    normalized_columns = {str(col).strip().lower(): col for col in df.columns}
    question_col = next(
        (normalized_columns[name] for name in QUESTION_COLUMNS if name in normalized_columns),
        df.columns[0],
    )
    answer_col = next(
        (normalized_columns[name] for name in ANSWER_COLUMNS if name in normalized_columns),
        None,
    )

    rows: list[QuestionRow] = []
    for idx, record in df.iterrows():
        raw_question: Any = record.get(question_col)
        if raw_question is None or str(raw_question).strip() == "":
            continue
        raw_reference = record.get(answer_col) if answer_col is not None else None
        reference = None
        if raw_reference is not None and str(raw_reference).strip().lower() != "nan":
            reference = str(raw_reference).strip()
        rows.append(QuestionRow(index=int(idx) + 1, question=str(raw_question).strip(), reference_answer=reference))

    return rows


def row_from_answer(row: QuestionRow, result: RagAnswer) -> dict[str, Any]:
    note = ""
    if row.reference_answer is None:
        note = "No reference answer column was provided, so Exact Match is N/A."

    return {
        "question_number": row.index,
        "question": row.question,
        "generated_answer": result.answer,
        "citations": result.citation_text,
        "top_document": result.top_source,
        "top_page": result.top_page,
        "recall_at_1": recall_at_1(row.question, result.top_source),
        "reference_answer": row.reference_answer or "",
        "exact_match": exact_match(result.answer, row.reference_answer),
        "hallucination_flag": hallucination_flag(result.answer, result.contexts),
        "notes": note,
    }


def _rate(values: list[str], positive_value: str = "True") -> str:
    scored = [value for value in values if value != "N/A"]
    if not scored:
        return "N/A"
    positives = sum(1 for value in scored if value == positive_value)
    return f"{positives / len(scored):.3f}"


def evaluate(config: RagConfig = DEFAULT_CONFIG) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:
        raise missing_dependency_error("pandas and openpyxl") from exc

    questions = read_questions(Path(config.questions_path))
    output_rows = []
    for question_row in questions:
        result = answer_question(question_row.question, config=config)
        output_rows.append(row_from_answer(question_row, result))

    recall_values = [str(row["recall_at_1"]) for row in output_rows]
    em_values = [str(row["exact_match"]) for row in output_rows]
    hallucination_reviews = [
        str(row["hallucination_flag"]) == "Needs review" for row in output_rows
    ]
    hallucination_rate = (
        f"{sum(hallucination_reviews) / len(hallucination_reviews):.3f}"
        if hallucination_reviews
        else "N/A"
    )

    output_path = Path(config.output_workbook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(output_rows).to_excel(writer, sheet_name="RAG Answers", index=False)
        summary = pd.DataFrame(
            [
                {"metric": "question_rows_processed", "value": len(output_rows)},
                {"metric": "recall_at_1_rate", "value": _rate(recall_values)},
                {"metric": "exact_match_rate", "value": _rate(em_values)},
                {"metric": "hallucination_review_rate", "value": hallucination_rate},
                {
                    "metric": "assignment_question_count_note",
                    "value": "RAG_Assignment.docx says 10 questions, but Questions.xlsx contains 9 non-header questions.",
                },
                {
                    "metric": "exact_match_note",
                    "value": "Exact Match is N/A unless Questions.xlsx includes a reference answer column.",
                },
                {
                    "metric": "recall_at_1_note",
                    "value": "Recall@1 is computed from NCT identifiers when a question includes one; otherwise N/A.",
                },
            ]
        )
        summary.to_excel(writer, sheet_name="Evaluation Notes", index=False)
    return output_path
