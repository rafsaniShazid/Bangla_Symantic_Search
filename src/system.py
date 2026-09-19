"""Load the corpus and expose the three retrieval methods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.data_loader import load_dataset
from src.hybrid_search import HybridSearcher
from src.tfidf_search import TFIDFSearcher
from src.word2vec_search import Word2VecSearcher, load_word2vec_model


METHOD_TFIDF = "tfidf"
METHOD_WORD2VEC = "word2vec"
METHOD_HYBRID = "hybrid"


class SemanticSearchSystem:
    def __init__(self, documents: pd.DataFrame, custom_model) -> None:
        if documents.empty:
            raise ValueError("The search system needs at least one document.")

        self.documents = documents.reset_index(drop=True).copy()
        self.tfidf = TFIDFSearcher().fit(self.documents)
        self.word2vec = Word2VecSearcher().fit(self.documents, custom_model)
        self.hybrid = HybridSearcher(self.tfidf, self.word2vec)

    def available_methods(self) -> list[str]:
        return [METHOD_TFIDF, METHOD_WORD2VEC, METHOD_HYBRID]

    def search(
        self,
        query: str,
        method: str = METHOD_TFIDF,
        top_k: int = config.DEFAULT_TOP_K,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        if method == METHOD_TFIDF:
            return self.tfidf.search(query, top_k)
        if method == METHOD_WORD2VEC:
            return self.word2vec.search(query, top_k)
        if method == METHOD_HYBRID:
            return self.hybrid.search(query, top_k, alpha)
        raise ValueError(f"Unknown retrieval method: {method}")


def build_search_system(
    data_path: str | Path = config.DEFAULT_DATA_PATH,
    custom_model_path: str | Path = config.CUSTOM_W2V_PATH,
) -> SemanticSearchSystem:
    """Build the search system from the news CSV and custom model file."""

    documents, _ = load_dataset(data_path)
    model = load_word2vec_model(custom_model_path)
    return SemanticSearchSystem(documents, model)
