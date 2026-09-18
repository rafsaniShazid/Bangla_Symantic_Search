"""Explainable TF-IDF retrieval for Bangla news documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


RESULT_COLUMNS = [
    "rank",
    "document_id",
    "title",
    "category",
    "similarity_score",
]


class TFIDFSearcher:
    """Rank news documents by cosine similarity to a TF-IDF query vector."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = config.TFIDF_NGRAM_RANGE,
        min_df: int | float = config.TFIDF_MIN_DF,
        max_df: int | float = config.TFIDF_MAX_DF,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.documents: pd.DataFrame | None = None
        self.document_matrix = None
        self.text_column: str | None = None

    def fit(
        self,
        documents: pd.DataFrame | Iterable[str],
        text_column: str = "text",
    ) -> "TFIDFSearcher":
        """Fit the vectorizer on a DataFrame or a sequence of document strings."""

        if isinstance(documents, pd.DataFrame):
            if text_column not in documents.columns:
                raise ValueError(f"Document text column not found: {text_column}")
            if not {"document_id", "title", "category"}.issubset(documents.columns):
                raise ValueError(
                    "DataFrame documents must contain document_id, title, and category."
                )
            self.documents = documents.reset_index(drop=True).copy()
            texts = self.documents[text_column].fillna("").astype(str).tolist()
            self.text_column = text_column
        else:
            texts = [str(document) for document in documents]
            self.documents = pd.DataFrame(
                {
                    "document_id": range(1, len(texts) + 1),
                    "title": texts,
                    "category": "",
                    text_column: texts,
                }
            )
            self.text_column = text_column

        if not texts or not any(text.strip() for text in texts):
            raise ValueError("Cannot fit TF-IDF on an empty document collection.")

        self.document_matrix = self.vectorizer.fit_transform(texts)
        return self

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
        """Return the highest-scoring documents for ``query``."""

        if self.documents is None or self.document_matrix is None: 
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query or not query.strip():
            return pd.DataFrame(columns=RESULT_COLUMNS)

        query_matrix = self.vectorizer.transform([query])
        scores = cosine_similarity(query_matrix, self.document_matrix).ravel()
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        results = self.documents.iloc[ranked_indices][
            ["document_id", "title", "category"]
        ].copy()
        results.insert(0, "rank", range(1, len(results) + 1))
        results["similarity_score"] = [float(scores[index]) for index in ranked_indices]
        return results[RESULT_COLUMNS].reset_index(drop=True)

    def save(self, directory: str | Path) -> None:
        """Persist the fitted searcher using scikit-learn's joblib helper."""

        if self.documents is None or self.document_matrix is None:
            raise RuntimeError("Call fit() before save().")
        from joblib import dump

        dump(self, Path(directory) / "tfidf_searcher.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "TFIDFSearcher":
        """Load a searcher saved by ``save``."""

        from joblib import load

        return load(Path(directory) / "tfidf_searcher.joblib")
