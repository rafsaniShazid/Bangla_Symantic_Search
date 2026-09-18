"""Explain how a query is interpreted by the retrieval models."""

from __future__ import annotations

from typing import Any

from src.preprocessing import preprocess_text, tokenize_bangla
from src.word2vec_search import get_keyed_vectors


def explain_tfidf_query(searcher, query: str) -> dict[str, Any]:
    """Return processed tokens and active TF-IDF features for one query."""

    processed_query = preprocess_text(query)
    tokens = processed_query.split()

    query_vector = searcher.vectorizer.transform([processed_query])
    feature_names = searcher.vectorizer.get_feature_names_out()

    active_features = [
        {
            "term": str(feature_names[index]),
            "weight": float(query_vector[0, index]),
        }
        for index in query_vector.nonzero()[1]
    ]
    active_features.sort(key=lambda item: (-item["weight"], item["term"]))

    active_terms = {item["term"] for item in active_features}
    matched_tokens = [token for token in tokens if token in active_terms]
    unmatched_tokens = [token for token in tokens if token not in searcher.vectorizer.vocabulary_]

    return {
        "processed_query": processed_query,
        "tokens": tokens,
        "matched_tokens": matched_tokens,
        "unmatched_tokens": unmatched_tokens,
        "active_features": active_features,
    }


def explain_word2vec_query(searcher, query: str) -> dict[str, Any]:
    """Return known/OOV query words and vocabulary coverage."""

    tokens = list(searcher.tokenizer(query))
    keyed_vectors = get_keyed_vectors(searcher.model)
    vocabulary = getattr(keyed_vectors, "key_to_index", {})

    known_words = [token for token in tokens if token in vocabulary]
    oov_words = [token for token in tokens if token not in vocabulary]

    coverage = 0.0
    if tokens:
        coverage = len(known_words) / len(tokens)

    return {
        "tokens": tokens,
        "known_words": known_words,
        "oov_words": oov_words,
        "coverage": coverage,
    }


def explain_method(system, query: str, method: str) -> dict[str, Any]:
    """Build the relevant explanation sections for a selected method."""

    explanation: dict[str, Any] = {
        "method": method,
        "query": query,
        "tfidf": None,
        "word2vec": None,
    }

    if method in {"tfidf", "hybrid_custom", "hybrid_pretrained"}:
        explanation["tfidf"] = explain_tfidf_query(system.tfidf, query)

    if method in {"custom_word2vec", "hybrid_custom"}:
        if system.custom_word2vec is None:
            raise RuntimeError("Custom Word2Vec model is not loaded.")
        explanation["word2vec"] = explain_word2vec_query(
            system.custom_word2vec,
            query,
        )

    if method in {"pretrained_word2vec", "hybrid_pretrained"}:
        if system.pretrained_word2vec is None:
            raise RuntimeError("Pretrained Word2Vec model is not loaded.")
        explanation["word2vec"] = explain_word2vec_query(
            system.pretrained_word2vec,
            query,
        )

    return explanation
