"""Mean-pooled Word2Vec retrieval for Bangla news documents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config


RESULT_COLUMNS = [
    "rank",
    "document_id",
    "title",
    "category",
    "similarity_score",
]
Tokenize = Callable[[str], Sequence[str]]


def _keyed_vectors(model):
    return getattr(model, "wv", model)


def _known_words(model, tokens: Iterable[str]) -> list[str]:
    keyed_vectors = _keyed_vectors(model)
    vocabulary = getattr(keyed_vectors, "key_to_index", None)
    if vocabulary is None:
        vocabulary = getattr(keyed_vectors, "vocab", {})
    return [token for token in tokens if token in vocabulary]


def get_document_vector(tokens: Iterable[str], model) -> np.ndarray | None:
    """Return a mean-pooled vector, or ``None`` when all tokens are OOV."""

    keyed_vectors = _keyed_vectors(model)
    known_words = _known_words(model, tokens)
    if not known_words:
        return None
    return np.mean([keyed_vectors[word] for word in known_words], axis=0)


def get_query_vector(tokens: Iterable[str], model) -> np.ndarray | None:
    """Return the mean-pooled query vector, or ``None`` when all tokens are OOV."""

    return get_document_vector(tokens, model)


class Word2VecSearcher:
    """Rank documents using cosine similarity between mean-pooled vectors."""

    def __init__(self, tokenizer: Tokenize | None = None) -> None:
        self.tokenizer = tokenizer or (lambda text: text.split())
        self.documents: pd.DataFrame | None = None
        self.model = None
        self.document_matrix: np.ndarray | None = None
        self.last_message = ""

    def fit(
        self,
        documents: pd.DataFrame,
        model,
        text_column: str = "text",
        token_column: str | None = None,
    ) -> "Word2VecSearcher":
        """Create document vectors from a DataFrame and a Word2Vec-like model."""

        required = {"document_id", "title", "category"}
        if not required.issubset(documents.columns):
            raise ValueError(
                "DataFrame documents must contain document_id, title, and category."
            )
        if token_column is None and text_column not in documents.columns:
            raise ValueError(f"Document text column not found: {text_column}")
        if token_column is not None and token_column not in documents.columns:
            raise ValueError(f"Document token column not found: {token_column}")

        self.documents = documents.reset_index(drop=True).copy()
        self.model = model
        vectors = []
        for _, document in self.documents.iterrows():
            tokens = (
                document[token_column]
                if token_column is not None
                else self.tokenizer(str(document[text_column]))
            )
            vector = get_document_vector(tokens, model)
            vectors.append(vector)

        dimensions = getattr(_keyed_vectors(model), "vector_size", None)
        if dimensions is None:
            first_vector = next((vector for vector in vectors if vector is not None), None)
            if first_vector is None:
                raise ValueError("No document contains a known Word2Vec token.")
            dimensions = len(first_vector)
        self.document_matrix = np.vstack(
            [vector if vector is not None else np.zeros(dimensions) for vector in vectors]
        )
        return self

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
        """Return ranked documents, or an empty result for an all-OOV query."""

        if self.documents is None or self.document_matrix is None or self.model is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query or not query.strip():
            self.last_message = "Enter a non-empty query."
            return pd.DataFrame(columns=RESULT_COLUMNS)

        query_vector = get_query_vector(self.tokenizer(query), self.model)
        if query_vector is None:
            self.last_message = "No query words were found in the Word2Vec vocabulary."
            return pd.DataFrame(columns=RESULT_COLUMNS)

        self.last_message = ""
        scores = cosine_similarity(query_vector.reshape(1, -1), self.document_matrix).ravel()
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


def load_word2vec_model(model_path: str | Path):
    """Load a native Gensim Word2Vec or KeyedVectors model from disk."""

    from gensim.models import KeyedVectors, Word2Vec

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Word2Vec model was not found: {path}")
    try:
        return Word2Vec.load(str(path))
    except Exception as word2vec_error:
        try:
            return KeyedVectors.load(str(path), mmap="r")
        except Exception as keyed_vectors_error:
            raise ValueError(
                f"Could not load a Gensim Word2Vec model from {path}."
            ) from keyed_vectors_error
