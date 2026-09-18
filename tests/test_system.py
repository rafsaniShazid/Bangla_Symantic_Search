"""Tests for the integrated retrieval system."""

import unittest

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.system import (
    METHOD_CUSTOM_W2V,
    METHOD_HYBRID_CUSTOM,
    METHOD_HYBRID_PRETRAINED,
    METHOD_PRETRAINED_W2V,
    METHOD_TFIDF,
    SemanticSearchSystem,
)


def build_tiny_vectors():
    vectors = KeyedVectors(vector_size=3)
    words = [
        "বাংলাদেশ",
        "অর্থনীতি",
        "বাজেট",
        "ক্রিকেট",
        "দল",
        "ম্যাচ",
        "প্রযুক্তি",
        "গবেষণা",
    ]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vectors.add_vectors(words, matrix)
    return vectors


class TestSemanticSearchSystem(unittest.TestCase):
    def setUp(self):
        self.documents = pd.DataFrame(
            {
                "document_id": ["n1", "n2", "n3"],
                "title": [
                    "বাংলাদেশের অর্থনীতি",
                    "ক্রিকেট দলের জয়",
                    "প্রযুক্তি গবেষণা",
                ],
                "content": [
                    "বাংলাদেশ অর্থনীতি বাজেট সংবাদ",
                    "ক্রিকেট দল ম্যাচ জিতেছে",
                    "প্রযুক্তি গবেষণা সংবাদ",
                ],
                "category": ["economy", "sports", "technology"],
                "text": [
                    "বাংলাদেশ অর্থনীতি বাজেট সংবাদ",
                    "ক্রিকেট দল ম্যাচ জিতেছে",
                    "প্রযুক্তি গবেষণা সংবাদ",
                ],
            }
        )
        self.vectors = build_tiny_vectors()

    def test_tfidf_is_always_available(self):
        system = SemanticSearchSystem(self.documents)

        self.assertEqual(system.available_methods(), [METHOD_TFIDF])

    def test_custom_model_enables_custom_and_hybrid_methods(self):
        system = SemanticSearchSystem(
            self.documents,
            custom_model=self.vectors,
        )

        self.assertEqual(
            system.available_methods(),
            [METHOD_TFIDF, METHOD_CUSTOM_W2V, METHOD_HYBRID_CUSTOM],
        )

    def test_both_word2vec_models_enable_all_methods(self):
        system = SemanticSearchSystem(
            self.documents,
            custom_model=self.vectors,
            pretrained_model=self.vectors,
        )

        self.assertEqual(
            system.available_methods(),
            [
                METHOD_TFIDF,
                METHOD_CUSTOM_W2V,
                METHOD_HYBRID_CUSTOM,
                METHOD_PRETRAINED_W2V,
                METHOD_HYBRID_PRETRAINED,
            ],
        )

    def test_tfidf_search_works_through_system(self):
        system = SemanticSearchSystem(self.documents)
        results = system.search(
            "অর্থনীতি বাজেট",
            method=METHOD_TFIDF,
            top_k=1,
        )

        self.assertEqual(results.iloc[0]["document_id"], "n1")

    def test_custom_word2vec_search_works_through_system(self):
        system = SemanticSearchSystem(
            self.documents,
            custom_model=self.vectors,
        )
        results = system.search(
            "ক্রিকেট ম্যাচ",
            method=METHOD_CUSTOM_W2V,
            top_k=1,
        )

        self.assertEqual(results.iloc[0]["document_id"], "n2")

    def test_pretrained_word2vec_search_works_through_system(self):
        system = SemanticSearchSystem(
            self.documents,
            pretrained_model=self.vectors,
        )
        results = system.search(
            "প্রযুক্তি গবেষণা",
            method=METHOD_PRETRAINED_W2V,
            top_k=1,
        )

        self.assertEqual(results.iloc[0]["document_id"], "n3")

    def test_hybrid_search_works_through_system(self):
        system = SemanticSearchSystem(
            self.documents,
            custom_model=self.vectors,
        )
        results = system.search(
            "অর্থনীতি বাজেট",
            method=METHOD_HYBRID_CUSTOM,
            top_k=2,
            alpha=0.6,
        )

        self.assertEqual(results.iloc[0]["document_id"], "n1")
        self.assertIn("hybrid_score", results.columns)

    def test_missing_model_method_has_clear_error(self):
        system = SemanticSearchSystem(self.documents)

        with self.assertRaisesRegex(RuntimeError, "Custom Word2Vec"):
            system.search(
                "ক্রিকেট",
                method=METHOD_CUSTOM_W2V,
            )

    def test_unknown_method_is_rejected(self):
        system = SemanticSearchSystem(self.documents)

        with self.assertRaises(ValueError):
            system.search("বাংলাদেশ", method="unknown")


if __name__ == "__main__":
    unittest.main()
