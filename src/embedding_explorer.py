"""Helpers for inspecting Word2Vec embedding neighborhoods."""

from __future__ import annotations

import pandas as pd

from src.word2vec_search import (
    model_summary,
    most_similar_words,
)


CUSTOM_EMBEDDING = "custom_word2vec"
PRETRAINED_EMBEDDING = "pretrained_word2vec"


def available_embedding_models(system) -> dict[str, object]:
    """Return the Word2Vec models that are currently loaded in the system."""

    models: dict[str, object] = {}

    if getattr(system, "custom_word2vec", None) is not None:
        models[CUSTOM_EMBEDDING] = system.custom_word2vec.model

    if getattr(system, "pretrained_word2vec", None) is not None:
        models[PRETRAINED_EMBEDDING] = system.pretrained_word2vec.model

    return models


def embedding_neighbors(
    model,
    word: str,
    topn: int = 10,
) -> pd.DataFrame:
    """Return nearest vocabulary words for one input word."""

    if topn < 1:
        raise ValueError("topn must be at least 1.")

    word = str(word).strip()
    if not word:
        return pd.DataFrame(columns=["word", "similarity"])

    rows = most_similar_words(model, word, topn=topn)

    return pd.DataFrame(
        [
            {
                "word": neighbor,
                "similarity": float(score),
            }
            for neighbor, score in rows
        ],
        columns=["word", "similarity"],
    )


def embedding_summary(model) -> dict[str, int]:
    """Return vocabulary size and vector dimension for one model."""

    return model_summary(model)
