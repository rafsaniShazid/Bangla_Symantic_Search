"""Load and validate the manually prepared retrieval-evaluation queries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


DEFAULT_QRELS_PATH = config.EVALUATION_DATA_DIR / "qrels.csv"
REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "query_type",
    "relevant_document_ids",
}
ALLOWED_QUERY_TYPES = {"lexical", "semantic", "mixed"}


def parse_relevant_document_ids(value: object) -> list[str]:
    """Parse a pipe-separated relevance judgement such as n1|n7|n12."""

    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    return [
        document_id.strip()
        for document_id in text.split("|")
        if document_id.strip()
    ]


def load_evaluation_queries(
    path: str | Path = DEFAULT_QRELS_PATH,
) -> pd.DataFrame:
    """Load the query set and validate its basic schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation query file was not found: {source}")

    queries = pd.read_csv(source, dtype=str, keep_default_na=False)

    missing = REQUIRED_COLUMNS - set(queries.columns)
    if missing:
        raise ValueError(
            "Evaluation file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    queries = queries[
        ["query_id", "query", "query_type", "relevant_document_ids"]
    ].copy()

    for column in ["query_id", "query", "query_type", "relevant_document_ids"]:
        queries[column] = queries[column].fillna("").astype(str).str.strip()

    if (queries["query_id"] == "").any():
        raise ValueError("Every evaluation query needs a query_id.")

    if queries["query_id"].duplicated().any():
        raise ValueError("Evaluation query_id values must be unique.")

    if (queries["query"] == "").any():
        raise ValueError("Every evaluation row needs a query.")

    invalid_types = set(queries["query_type"]) - ALLOWED_QUERY_TYPES
    if invalid_types:
        raise ValueError(
            "Unsupported query_type values: "
            + ", ".join(sorted(invalid_types))
        )

    queries["relevant_ids"] = queries["relevant_document_ids"].map(
        parse_relevant_document_ids
    )
    return queries


def count_completed_judgements(queries: pd.DataFrame) -> int:
    """Count queries that already have at least one relevant document ID."""

    if "relevant_ids" not in queries.columns:
        raise ValueError("Expected a relevant_ids column.")

    return int(queries["relevant_ids"].map(bool).sum())
