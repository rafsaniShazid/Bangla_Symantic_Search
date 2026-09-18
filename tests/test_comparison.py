"""Tests for model-comparison utilities."""

import unittest

import pandas as pd

from src.comparison import compare_methods


class FakeSystem:
    def available_methods(self):
        return ["tfidf", "custom_word2vec"]

    def search(self, query, method, top_k=5, alpha=None):
        if method == "tfidf":
            return pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "document_id": "n1",
                        "title": "Budget",
                        "category": "economy",
                        "similarity_score": 0.80,
                    },
                    {
                        "rank": 2,
                        "document_id": "n2",
                        "title": "Market",
                        "category": "economy",
                        "similarity_score": 0.40,
                    },
                ]
            ).head(top_k)

        return pd.DataFrame(
            [
                {
                    "rank": 1,
                    "document_id": "n2",
                    "title": "Market",
                    "category": "economy",
                    "similarity_score": 0.90,
                },
                {
                    "rank": 2,
                    "document_id": "n1",
                    "title": "Budget",
                    "category": "economy",
                    "similarity_score": 0.60,
                },
            ]
        ).head(top_k)


class HybridFakeSystem(FakeSystem):
    def available_methods(self):
        return ["hybrid_custom"]

    def search(self, query, method, top_k=5, alpha=None):
        return pd.DataFrame(
            [
                {
                    "rank": 1,
                    "document_id": "n1",
                    "title": "Budget",
                    "category": "economy",
                    "hybrid_score": 0.77,
                    "tfidf_score": 0.8,
                    "word2vec_score": 0.74,
                }
            ]
        )


class TestComparison(unittest.TestCase):
    def test_compare_methods_runs_all_available_methods(self):
        comparison = compare_methods(
            FakeSystem(),
            query="economy",
            top_k=2,
        )

        self.assertEqual(
            comparison["method"].unique().tolist(),
            ["tfidf", "custom_word2vec"],
        )
        self.assertEqual(len(comparison), 4)

    def test_comparison_uses_similarity_score_for_normal_searchers(self):
        comparison = compare_methods(
            FakeSystem(),
            query="economy",
            methods=["tfidf"],
            top_k=1,
        )

        self.assertAlmostEqual(comparison.iloc[0]["score"], 0.80)

    def test_comparison_uses_hybrid_score_for_hybrid_searcher(self):
        comparison = compare_methods(
            HybridFakeSystem(),
            query="economy",
            top_k=1,
        )

        self.assertAlmostEqual(comparison.iloc[0]["score"], 0.77)

    def test_empty_query_returns_empty_table(self):
        comparison = compare_methods(FakeSystem(), query="   ")

        self.assertTrue(comparison.empty)

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_methods(FakeSystem(), query="economy", top_k=0)


if __name__ == "__main__":
    unittest.main()
