"""Search news using a custom Word2Vec model and mean-pooled vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.preprocessing import tokenize_bangla


RESULT_COLUMNS = ["rank", "document_id", "title", "category", "similarity_score"]


def get_document_vector(tokens, model) -> np.ndarray | None:
    """Average vectors for known words; return None when every word is OOV."""

    vectors = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


class Word2VecSearcher:
    def __init__(self) -> None:
        self.documents: pd.DataFrame | None = None
        self.model = None
        self.document_matrix: np.ndarray | None = None
        self.last_message = ""

    def fit(self, documents: pd.DataFrame, model) -> "Word2VecSearcher":
        """Build one mean-pooled vector for each news article."""

        required = {"document_id", "title", "category", "text"}
        if not required.issubset(documents.columns):
            raise ValueError("Documents need document_id, title, category, and text.")
        if documents.empty:
            raise ValueError("Cannot fit Word2Vec on an empty document collection.")

        self.documents = documents.reset_index(drop=True).copy()
        self.model = model
        rows = []
        for text in self.documents["text"].fillna("").astype(str):
            vector = get_document_vector(tokenize_bangla(text), model)
            if vector is None:
                vector = np.zeros(model.wv.vector_size)
            rows.append(vector)
        self.document_matrix = np.vstack(rows)
        return self

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
        """Rank documents by cosine similarity to the pooled query vector."""

        if self.documents is None or self.document_matrix is None or self.model is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = get_document_vector(tokenize_bangla(query), self.model)
        if query_vector is None:
            self.last_message = "No query words were found in the Word2Vec vocabulary."
            return pd.DataFrame(columns=RESULT_COLUMNS)

        self.last_message = ""
        scores = cosine_similarity(
            query_vector.reshape(1, -1), self.document_matrix
        ).ravel()
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        results = self.documents.iloc[indices][
            ["document_id", "title", "category"]
        ].copy()
        results.insert(0, "rank", range(1, len(results) + 1))
        results["similarity_score"] = [float(scores[i]) for i in indices]
        return results[RESULT_COLUMNS].reset_index(drop=True)


def load_word2vec_model(model_path: str | Path):
    """Load only a custom Gensim Word2Vec .model file."""

    from gensim.models import Word2Vec

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Custom Word2Vec model was not found: {path}")
    if path.suffix.lower() != ".model":
        raise ValueError("Custom Word2Vec model must be a .model file.")
    return Word2Vec.load(str(path))
