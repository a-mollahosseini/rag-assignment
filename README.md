# RAG Assignment

This project implements the homework RAG pipeline for the provided clinical-trial PDFs.
The main code is under `src/rag_assignment/`.

## What It Does

- Loads the 20 PDF files from `data/`.
- Splits pages into overlapping chunks.
- Embeds chunks with Ollama model `mxbai-embed-large`.
- Stores vectors in a persistent Chroma database.
- Retrieves relevant chunks for each question.
- Generates concise cited answers with Ollama model `mistral`.
- Exports an Excel workbook with answers, citations, Recall@1, Exact Match status, and hallucination review flags.

The assignment document mentions 10 questions, but `Questions.xlsx` contains 9 non-header questions. This project processes the 9 questions that are actually present in the workbook.

## Local Setup

Install Ollama first, then pull the required models:

```bash
ollama pull mxbai-embed-large
ollama pull mistral
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Build the vector database:

```bash
python -m rag_assignment.cli ingest --reset
```

Ask one question:

```bash
python -m rag_assignment.cli ask "In the Access-H2O smart faucet feasibility study, how many subjects will be recruited?"
```

Run all questions and export the answer workbook:

```bash
python -m rag_assignment.cli evaluate
```

The generated workbook is written to:

```text
outputs/rag_answers.xlsx
```

## Google Colab

Open `RAG_Assignment_Colab.ipynb` in Google Colab. The notebook clones this GitHub repository, installs dependencies, starts Ollama, pulls the required models, runs ingestion, and exports the answer workbook.

If the repo is private, authenticate Colab with a GitHub token before cloning.

## CLI Reference

```bash
python -m rag_assignment.cli ingest --reset
python -m rag_assignment.cli ask "your question here"
python -m rag_assignment.cli evaluate
```

Useful options:

- `--data-dir`: source PDF directory.
- `--questions-path`: path to `Questions.xlsx`.
- `--chroma-dir`: Chroma persistence directory.
- `--output-workbook`: Excel output path.
- `--embedding-model`: Ollama embedding model name.
- `--generation-model`: Ollama generation model name.
- `--top-k`: number of retrieved chunks.

## Metrics Notes

- `Recall@1` is computed when a question contains an `NCT########` identifier. If no trial ID appears in the question, the metric is marked `N/A`.
- `Exact Match` requires a reference-answer column in the question workbook. The provided workbook does not include reference answers, so EM is marked `N/A`.
- The hallucination flag is a lightweight review heuristic. It marks answers for review when the model says evidence is missing or when too many answer terms are absent from retrieved context.
