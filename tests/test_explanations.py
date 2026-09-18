"""Tests for search-explanation helpers."""

import unittest

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.bangla_tfidf_search import BanglaTFIDFSearcher
from src.explanations import (
    explain_method,
    explain_tfidf_query,
    explain_word2vec_query,
)
from src.word2vec_search import Word2VecSearcher


def build_vectors():
    vectors = KeyedVectors(vector_size=3)
    words = ["অর্থনীতি", "বাজেট", "ক্রিকেট"]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors.add_vectors(words, matrix)
    return vectors


class DummySystem:
    def __init__(self, tfidf, custom=None, pretrained=None):
        self.tfidf = tfidf
        self.custom_word2vec = custom
        self.pretrained_word2vec = pretrained


class TestSearchExplanations(unittest.TestCase):
    def setUp(self):
        self.documents = pd.DataFrame(
            {
                "document_id": ["n1", "n2"],
                "title": ["অর্থনীতি", "ক্রিকেট"],
                "category": ["economy", "sports"],
                "text": [
                    "বাংলাদেশ অর্থনীতি বাজেট",
                    "বাংলাদেশ ক্রিকেট ম্যাচ",
                ],
            }
        )
        self.tfidf = BanglaTFIDFSearcher().fit(self.documents)

        self.word2vec = Word2VecSearcher(
            tokenizer=lambda text: text.split()
        ).fit(
            self.documents,
            build_vectors(),
            text_column="text",
        )

    def test_tfidf_explanation_reports_active_features(self):
        details = explain_tfidf_query(
            self.tfidf,
            "এই অর্থনীতি এবং বাজেট",
        )

        self.assertEqual(details["tokens"], ["অর্থনীতি", "বাজেট"])
        self.assertIn("অর্থনীতি", details["matched_tokens"])
        self.assertIn("বাজেট", details["matched_tokens"])
        self.assertTrue(details["active_features"])

    def test_tfidf_explanation_reports_unknown_token(self):
        details = explain_tfidf_query(
            self.tfidf,
            "অর্থনীতি অজানা",
        )

        self.assertIn("অজানা", details["unmatched_tokens"])

    def test_word2vec_explanation_reports_known_and_oov_words(self):
        details = explain_word2vec_query(
            self.word2vec,
            "অর্থনীতি অজানা",
        )

        self.assertEqual(details["known_words"], ["অর্থনীতি"])
        self.assertEqual(details["oov_words"], ["অজানা"])
        self.assertAlmostEqual(details["coverage"], 0.5)

    def test_method_explanation_combines_hybrid_sections(self):
        system = DummySystem(
            tfidf=self.tfidf,
            custom=self.word2vec,
        )

        details = explain_method(
            system,
            "অর্থনীতি বাজেট",
            "hybrid_custom",
        )

        self.assertIsNotNone(details["tfidf"])
        self.assertIsNotNone(details["word2vec"])

    def test_method_explanation_for_tfidf_has_no_word2vec_section(self):
        system = DummySystem(tfidf=self.tfidf)

        details = explain_method(
            system,
            "অর্থনীতি",
            "tfidf",
        )

        self.assertIsNotNone(details["tfidf"])
        self.assertIsNone(details["word2vec"])


if __name__ == "__main__":
    unittest.main()
