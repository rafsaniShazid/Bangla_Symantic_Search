"""Evaluation metrics for ranked news retrieval results."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean


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
