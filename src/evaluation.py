"""Evaluation metrics for ranked news retrieval results."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from statistics import mean

import pandas as pd

import config


def _validate_k(k: int) -> int:
    if int(k) < 1:
        raise ValueError("k must be at least 1.")
    return int(k)


def _normalise_ids(values: Iterable[object]) -> list[str]:
    """Convert document IDs to comparable strings while preserving order."""

    return [str(value) for value in values]


def precision_at_k(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 10,
) -> float:
    """Return the fraction of the first k results that are relevant."""

    k = _validate_k(k)
    retrieved = _normalise_ids(retrieved_ids)[:k]
    relevant = set(_normalise_ids(relevant_ids))

    hits = sum(document_id in relevant for document_id in retrieved)
    return hits / k


def recall_at_k(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 10,
) -> float:
    """Return the fraction of known relevant documents found in the first k."""

    k = _validate_k(k)
    retrieved = set(_normalise_ids(retrieved_ids)[:k])
    relevant = set(_normalise_ids(relevant_ids))

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
) -> float:
    """Return 1/rank for the first relevant result, or 0 when none is found."""

    relevant = set(_normalise_ids(relevant_ids))

    for rank, document_id in enumerate(_normalise_ids(retrieved_ids), start=1):
        if document_id in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_query(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 10,
) -> dict[str, float]:
    """Calculate Precision@K, Recall@K, and reciprocal rank for one query."""

    retrieved = _normalise_ids(retrieved_ids)
    relevant = _normalise_ids(relevant_ids)

    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k=k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k=k),
        "reciprocal_rank": reciprocal_rank(retrieved, relevant),
    }


def mean_evaluation(
    rankings: Iterable[tuple[Iterable[object], Iterable[object]]],
    k: int = 10,
) -> dict[str, float]:
    """Average retrieval metrics across several evaluation queries."""

    k = _validate_k(k)
    query_metrics = [
        evaluate_query(retrieved, relevant, k=k)
        for retrieved, relevant in rankings
    ]

    if not query_metrics:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            "mrr": 0.0,
        }

    return {
        f"precision@{k}": mean(
            metrics[f"precision@{k}"] for metrics in query_metrics
        ),
        f"recall@{k}": mean(
            metrics[f"recall@{k}"] for metrics in query_metrics
        ),
        "mrr": mean(
            metrics["reciprocal_rank"] for metrics in query_metrics
        ),
    }


def load_evaluation_queries(
    path: str | Path = config.EVALUATION_DATA_DIR / "qrels.csv",
) -> pd.DataFrame:
    """Load prepared queries and their manually judged document IDs."""

    queries = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = ["query_id", "query", "query_type", "relevant_document_ids"]
    if not set(columns).issubset(queries.columns):
        raise ValueError(
            "Evaluation CSV needs query_id, query, query_type, "
            "and relevant_document_ids."
        )
    queries = queries[columns].copy()
    for column in columns:
        queries[column] = queries[column].str.strip()
    if queries["query_id"].eq("").any() or queries["query_id"].duplicated().any():
        raise ValueError("Evaluation query IDs must be nonempty and unique.")
    if queries["query"].eq("").any():
        raise ValueError("Evaluation queries must be nonempty.")

    queries["relevant_ids"] = queries["relevant_document_ids"].map(
        lambda value: [item.strip() for item in value.split("|") if item.strip()]
    )
    return queries


def evaluate_system(
    system, queries: pd.DataFrame, k: int = 5, alpha: float = 0.5
) -> pd.DataFrame:
    """Average metrics for each method over queries with real judgements."""

    _validate_k(k)
    judged = queries[queries["relevant_ids"].map(bool)]
    rows = []
    for method in system.available_methods():
        rankings = []
        for _, query in judged.iterrows():
            results = system.search(
                query["query"], method=method, top_k=k, alpha=alpha
            )
            rankings.append((results["document_id"], query["relevant_ids"]))
        if rankings:
            rows.append(
                {"method": method, "queries": len(rankings), **mean_evaluation(rankings, k)}
            )
    return pd.DataFrame(
        rows,
        columns=["method", "queries", f"precision@{k}", f"recall@{k}", "mrr"],
    )
