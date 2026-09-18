"""Run retrieval evaluation over manually judged Bangla queries."""

from __future__ import annotations

import pandas as pd

from src.evaluation import evaluate_query


HYBRID_METHODS = {
    "hybrid_custom",
    "hybrid_pretrained",
}


def completed_queries(queries: pd.DataFrame) -> pd.DataFrame:
    """Return only queries that already have relevance judgements."""

    if "relevant_ids" not in queries.columns:
        raise ValueError("Expected a relevant_ids column.")

    mask = queries["relevant_ids"].map(bool)
    return queries.loc[mask].reset_index(drop=True).copy()


def evaluate_method(
    system,
    queries: pd.DataFrame,
    method: str,
    k: int = 5,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Evaluate one retrieval method on all judged queries."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    judged = completed_queries(queries)
    rows: list[dict[str, object]] = []

    for _, query_row in judged.iterrows():
        results = system.search(
            query=query_row["query"],
            method=method,
            top_k=k,
            alpha=alpha if method in HYBRID_METHODS else None,
        )

        retrieved_ids = (
            results["document_id"].astype(str).tolist()
            if "document_id" in results.columns
            else []
        )

        metrics = evaluate_query(
            retrieved_ids=retrieved_ids,
            relevant_ids=query_row["relevant_ids"],
            k=k,
        )

        rows.append(
            {
                "method": method,
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "query_type": query_row["query_type"],
                f"precision@{k}": metrics[f"precision@{k}"],
                f"recall@{k}": metrics[f"recall@{k}"],
                "reciprocal_rank": metrics["reciprocal_rank"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_methods(
    system,
    queries: pd.DataFrame,
    methods: list[str] | None = None,
    k: int = 5,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Evaluate several available retrieval methods."""

    selected_methods = methods or system.available_methods()
    frames = [
        evaluate_method(
            system=system,
            queries=queries,
            method=method,
            k=k,
            alpha=alpha,
        )
        for method in selected_methods
    ]

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "method",
                "query_id",
                "query",
                "query_type",
                f"precision@{k}",
                f"recall@{k}",
                "reciprocal_rank",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def summarize_evaluation(
    details: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Return one summary row per retrieval method."""

    precision_column = f"precision@{k}"
    recall_column = f"recall@{k}"

    if details.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "queries",
                precision_column,
                recall_column,
                "mrr",
            ]
        )

    grouped = (
        details.groupby("method", sort=False)
        .agg(
            queries=("query_id", "count"),
            **{
                precision_column: (precision_column, "mean"),
                recall_column: (recall_column, "mean"),
                "mrr": ("reciprocal_rank", "mean"),
            },
        )
        .reset_index()
    )

    return grouped


def summarize_by_query_type(
    details: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Return method performance split by lexical/semantic/mixed queries."""

    precision_column = f"precision@{k}"
    recall_column = f"recall@{k}"

    if details.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "query_type",
                "queries",
                precision_column,
                recall_column,
                "mrr",
            ]
        )

    grouped = (
        details.groupby(["method", "query_type"], sort=False)
        .agg(
            queries=("query_id", "count"),
            **{
                precision_column: (precision_column, "mean"),
                recall_column: (recall_column, "mean"),
                "mrr": ("reciprocal_rank", "mean"),
            },
        )
        .reset_index()
    )

    return grouped
