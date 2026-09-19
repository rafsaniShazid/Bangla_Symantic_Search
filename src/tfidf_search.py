"""Bangla TF-IDF search with cosine similarity."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.preprocessing import preprocess_text


RESULT_COLUMNS = ["rank", "document_id", "title", "category", "similarity_score"]


class TFIDFSearcher:
    def __init__(
        self,
        ngram_range: tuple[int, int] = config.TFIDF_NGRAM_RANGE,
        min_df: int | float = config.TFIDF_MIN_DF,
        max_df: int | float = config.TFIDF_MAX_DF,
    ) -> None:
        # The shared preprocessor separates whole Bangla words with spaces.
        # sklearn's default word pattern can split Bangla combining characters.
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
        )
        self.documents: pd.DataFrame | None = None
        self.document_matrix = None

    def fit(self, documents: pd.DataFrame) -> "TFIDFSearcher":
        """Preprocess and vectorize the news articles."""

        required = {"document_id", "title", "category", "text"}
        if not required.issubset(documents.columns):
            raise ValueError("Documents need document_id, title, category, and text.")

        texts = documents["text"].fillna("").astype(str).map(preprocess_text)
        if not texts.map(bool).any():
            raise ValueError("Cannot fit TF-IDF on an empty document collection.")

        self.documents = documents.reset_index(drop=True).copy()
        self.document_matrix = self.vectorizer.fit_transform(texts)
        return self

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
        """Rank documents by query-to-document cosine similarity."""

        if self.documents is None or self.document_matrix is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        processed_query = preprocess_text(query)
        if not processed_query:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        query_vector = self.vectorizer.transform([processed_query])
        if query_vector.nnz == 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        scores = cosine_similarity(query_vector, self.document_matrix).ravel()
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        results = self.documents.iloc[indices][
            ["document_id", "title", "category"]
        ].copy()
        results.insert(0, "rank", range(1, len(results) + 1))
        results["similarity_score"] = [float(scores[i]) for i in indices]
        return results[RESULT_COLUMNS].reset_index(drop=True)
