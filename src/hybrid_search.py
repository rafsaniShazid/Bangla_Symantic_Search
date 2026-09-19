"""Combine TF-IDF and custom Word2Vec scores for the same documents."""

from __future__ import annotations

import pandas as pd

import config


RESULT_COLUMNS = [
    "rank", "document_id", "title", "category",
    "hybrid_score", "tfidf_score", "word2vec_score",
]


def combine_rankings(
    tfidf_results: pd.DataFrame,
    word2vec_results: pd.DataFrame,
    alpha: float = 0.5,
    top_k: int = config.DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Align results by document ID, then apply the weighted score formula."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if tfidf_results.empty and word2vec_results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # If one query vector is unavailable, use the other score directly.
    if tfidf_results.empty:
        alpha = 0.0
    elif word2vec_results.empty:
        alpha = 1.0

    available = [table for table in (tfidf_results, word2vec_results) if not table.empty]
    combined = pd.concat(available, ignore_index=True)
    combined = combined.drop_duplicates("document_id").copy()
    tfidf_scores = tfidf_results.set_index("document_id")["similarity_score"]
    word2vec_scores = word2vec_results.set_index("document_id")["similarity_score"]
    combined["tfidf_score"] = (
        combined["document_id"].map(tfidf_scores).fillna(0.0).astype(float)
    )
    combined["word2vec_score"] = (
        combined["document_id"].map(word2vec_scores).fillna(0.0).astype(float)
    )
    combined["hybrid_score"] = (
        alpha * combined["tfidf_score"]
        + (1 - alpha) * combined["word2vec_score"]
    )
    combined = combined.sort_values("hybrid_score", ascending=False, kind="stable")
    combined = combined.head(top_k).reset_index(drop=True)
    combined["rank"] = range(1, len(combined) + 1)
    return combined[RESULT_COLUMNS]


class HybridSearcher:
    def __init__(self, tfidf_searcher, word2vec_searcher) -> None:
        self.tfidf_searcher = tfidf_searcher
        self.word2vec_searcher = word2vec_searcher

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        """Score the full corpus with both methods before selecting Top-K."""

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query or not query.strip():
            return pd.DataFrame(columns=RESULT_COLUMNS)

        candidate_count = len(self.tfidf_searcher.documents)
        tfidf_results = self.tfidf_searcher.search(query, top_k=candidate_count)
        word2vec_results = self.word2vec_searcher.search(query, top_k=candidate_count)
        return combine_rankings(tfidf_results, word2vec_results, alpha, top_k)
