"""Hybrid retrieval that combines TF-IDF and Word2Vec scores."""

from __future__ import annotations

import pandas as pd

import config


HYBRID_RESULT_COLUMNS = [
    "rank",
    "document_id",
    "title",
    "category",
    "hybrid_score",
    "tfidf_score",
    "word2vec_score",
]


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    return value


def combine_rankings(
    tfidf_results: pd.DataFrame,
    word2vec_results: pd.DataFrame,
    alpha: float = 0.5,
    top_k: int = config.DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Combine two ranked result tables using a weighted score.

    alpha controls the lexical TF-IDF contribution:
        hybrid = alpha * tfidf + (1 - alpha) * word2vec

    If one retrieval method cannot return any result (for example an all-OOV
    Word2Vec query), the available method is used as a clean fallback.
    """

    alpha = _validate_alpha(alpha)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if tfidf_results.empty and word2vec_results.empty:
        return pd.DataFrame(columns=HYBRID_RESULT_COLUMNS)

    if tfidf_results.empty:
        fallback = word2vec_results.copy()
        fallback["tfidf_score"] = 0.0
        fallback["word2vec_score"] = fallback["similarity_score"].astype(float)
        fallback["hybrid_score"] = fallback["word2vec_score"]
        fallback = fallback.sort_values(
            ["hybrid_score", "document_id"],
            ascending=[False, True],
            kind="stable",
        ).head(top_k)
        fallback = fallback.reset_index(drop=True)
        fallback["rank"] = range(1, len(fallback) + 1)
        return fallback[HYBRID_RESULT_COLUMNS]

    if word2vec_results.empty:
        fallback = tfidf_results.copy()
        fallback["tfidf_score"] = fallback["similarity_score"].astype(float)
        fallback["word2vec_score"] = 0.0
        fallback["hybrid_score"] = fallback["tfidf_score"]
        fallback = fallback.sort_values(
            ["hybrid_score", "document_id"],
            ascending=[False, True],
            kind="stable",
        ).head(top_k)
        fallback = fallback.reset_index(drop=True)
        fallback["rank"] = range(1, len(fallback) + 1)
        return fallback[HYBRID_RESULT_COLUMNS]

    lexical = tfidf_results[
        ["document_id", "title", "category", "similarity_score"]
    ].rename(columns={"similarity_score": "tfidf_score"})

    semantic = word2vec_results[
        ["document_id", "similarity_score"]
    ].rename(columns={"similarity_score": "word2vec_score"})

    combined = lexical.merge(
        semantic,
        on="document_id",
        how="outer",
    )
    combined["tfidf_score"] = combined["tfidf_score"].fillna(0.0).astype(float)
    combined["word2vec_score"] = (
        combined["word2vec_score"].fillna(0.0).astype(float)
    )

    # Metadata normally comes from the TF-IDF result set because both searchers
    # use the same document collection.
    combined["title"] = combined["title"].fillna("")
    combined["category"] = combined["category"].fillna("")

    combined["hybrid_score"] = (
        alpha * combined["tfidf_score"]
        + (1.0 - alpha) * combined["word2vec_score"]
    )

    combined = combined.sort_values(
        ["hybrid_score", "document_id"],
        ascending=[False, True],
        kind="stable",
    ).head(top_k)
    combined = combined.reset_index(drop=True)
    combined.insert(0, "rank", range(1, len(combined) + 1))

    return combined[HYBRID_RESULT_COLUMNS]


class HybridSearcher:
    """Run lexical and semantic retrieval, then combine their scores."""

    def __init__(
        self,
        tfidf_searcher,
        word2vec_searcher,
        alpha: float = 0.5,
    ) -> None:
        self.tfidf_searcher = tfidf_searcher
        self.word2vec_searcher = word2vec_searcher
        self.alpha = _validate_alpha(alpha)

    def _candidate_count(self, top_k: int) -> int:
        counts = [top_k]

        for searcher in (self.tfidf_searcher, self.word2vec_searcher):
            documents = getattr(searcher, "documents", None)
            if documents is not None:
                counts.append(len(documents))

        return max(counts)

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Return top hybrid results for a query."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query or not query.strip():
            return pd.DataFrame(columns=HYBRID_RESULT_COLUMNS)

        weight = self.alpha if alpha is None else _validate_alpha(alpha)
        candidate_count = self._candidate_count(top_k)

        tfidf_results = self.tfidf_searcher.search(
            query,
            top_k=candidate_count,
        )
        word2vec_results = self.word2vec_searcher.search(
            query,
            top_k=candidate_count,
        )

        return combine_rankings(
            tfidf_results,
            word2vec_results,
            alpha=weight,
            top_k=top_k,
        )
