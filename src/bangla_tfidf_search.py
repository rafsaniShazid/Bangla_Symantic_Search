"""Bangla-aware adapter around the existing TF-IDF searcher."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.preprocessing import preprocess_text
from src.tfidf_search import TFIDFSearcher


class BanglaTFIDFSearcher(TFIDFSearcher):
    """Use shared Bangla preprocessing while reusing the existing ranking logic."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = config.TFIDF_NGRAM_RANGE,
        min_df: int | float = config.TFIDF_MIN_DF,
        max_df: int | float = config.TFIDF_MAX_DF,
    ) -> None:
        super().__init__(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

        # The shared preprocessor already creates whitespace-separated tokens.
        # Splitting on whitespace avoids the generic \w+ behavior that can
        # break Bangla words around combining characters.
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            tokenizer=str.split,
            token_pattern=None,
            preprocessor=None,
            lowercase=False,
        )

    def fit(
        self,
        documents: pd.DataFrame | Iterable[str],
        text_column: str = "text",
    ) -> "BanglaTFIDFSearcher":
        """Preprocess the corpus, then fit the inherited TF-IDF pipeline."""

        if isinstance(documents, pd.DataFrame):
            if text_column not in documents.columns:
                raise ValueError(f"Document text column not found: {text_column}")

            prepared = documents.copy()
            prepared["_tfidf_text"] = (
                prepared[text_column]
                .fillna("")
                .astype(str)
                .map(preprocess_text)
            )
            super().fit(prepared, text_column="_tfidf_text")
        else:
            prepared_texts = [preprocess_text(str(document)) for document in documents]
            super().fit(prepared_texts, text_column="_tfidf_text")

        return self

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
        """Preprocess a Bangla query before using inherited cosine ranking."""

        processed_query = preprocess_text(query)
        return super().search(processed_query, top_k=top_k)
