"""Utilities for comparing retrieval methods on the same query."""

from __future__ import annotations

import pandas as pd

import config


def _result_score_column(results: pd.DataFrame) -> str | None:
    """Return the main score column used by a retrieval result table."""

    if "hybrid_score" in results.columns:
        return "hybrid_score"
    if "similarity_score" in results.columns:
        return "similarity_score"
    return None


def compare_methods(
    system,
    query: str,
    methods: list[str] | None = None,
    top_k: int = 5,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Run the same query through several retrieval methods.

    The returned table uses one consistent schema so the Streamlit UI can
    compare lexical, semantic, and hybrid rankings directly.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not query or not query.strip():
        return pd.DataFrame(
            columns=[
                "method",
                "rank",
                "document_id",
                "title",
                "category",
                "score",
            ]
        )

    selected_methods = methods or system.available_methods()
    rows: list[dict[str, object]] = []

    for method in selected_methods:
        results = system.search(
            query=query,
            method=method,
            top_k=top_k,
            alpha=alpha,
        )

        score_column = _result_score_column(results)

        for _, row in results.iterrows():
            score = (
                float(row[score_column])
                if score_column is not None
                else 0.0
            )

            rows.append(
                {
                    "method": method,
                    "rank": int(row["rank"]),
                    "document_id": str(row["document_id"]),
                    "title": str(row["title"]),
                    "category": str(row["category"]),
                    "score": score,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "rank",
            "document_id",
            "title",
            "category",
            "score",
        ],
    )
