"""Tests for the Word2Vec embedding explorer."""

import unittest

import numpy as np
from gensim.models import KeyedVectors

from src.embedding_explorer import (
    CUSTOM_EMBEDDING,
    PRETRAINED_EMBEDDING,
    available_embedding_models,
    embedding_neighbors,
    embedding_summary,
)


def build_vectors():
    vectors = KeyedVectors(vector_size=3)
    words = ["অর্থনীতি", "বাজেট", "বাজার", "ক্রিকেট"]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.90, 0.10, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors.add_vectors(words, matrix)
    return vectors


class DummySearcher:
    def __init__(self, model):
        self.model = model


class DummySystem:
    def __init__(self, custom=None, pretrained=None):
        self.custom_word2vec = (
            DummySearcher(custom) if custom is not None else None
        )
        self.pretrained_word2vec = (
            DummySearcher(pretrained) if pretrained is not None else None
        )


class TestEmbeddingExplorer(unittest.TestCase):
    def test_available_models_reports_loaded_word2vec_models(self):
        vectors = build_vectors()
        system = DummySystem(custom=vectors, pretrained=vectors)

        models = available_embedding_models(system)

        self.assertEqual(
            list(models.keys()),
            [CUSTOM_EMBEDDING, PRETRAINED_EMBEDDING],
        )

    def test_available_models_can_be_empty(self):
        self.assertEqual(available_embedding_models(DummySystem()), {})

    def test_neighbors_returns_ranked_similar_words(self):
        neighbors = embedding_neighbors(
            build_vectors(),
            "অর্থনীতি",
            topn=2,
        )

        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors.iloc[0]["word"], "বাজেট")
        self.assertGreater(
            neighbors.iloc[0]["similarity"],
            neighbors.iloc[1]["similarity"],
        )

    def test_oov_word_returns_empty_table(self):
        neighbors = embedding_neighbors(
            build_vectors(),
            "অজানা",
            topn=3,
        )

        self.assertTrue(neighbors.empty)

    def test_empty_word_returns_empty_table(self):
        self.assertTrue(
            embedding_neighbors(build_vectors(), "   ").empty
        )

    def test_summary_reports_vocabulary_and_dimension(self):
        summary = embedding_summary(build_vectors())

        self.assertEqual(summary["vocabulary_size"], 4)
        self.assertEqual(summary["vector_size"], 3)

    def test_invalid_topn_is_rejected(self):
        with self.assertRaises(ValueError):
            embedding_neighbors(build_vectors(), "অর্থনীতি", topn=0)


if __name__ == "__main__":
    unittest.main()
