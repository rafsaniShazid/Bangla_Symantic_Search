"""Tests for hybrid lexical + semantic retrieval."""

import unittest

import pandas as pd

from src.hybrid_search import HybridSearcher, combine_rankings


def _results(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "rank",
            "document_id",
            "title",
            "category",
            "similarity_score",
        ],
    )


class StaticSearcher:
    def __init__(self, results):
        self.results = results
        self.documents = pd.DataFrame(
            {"document_id": results["document_id"].tolist()}
        )

    def search(self, query, top_k=10):
        return self.results.head(top_k).copy()


class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        self.tfidf = _results(
            [
                [1, 1, "Budget", "economy", 0.90],
                [2, 2, "Market", "economy", 0.50],
                [3, 3, "Cricket", "sports", 0.10],
            ]
        )
        self.word2vec = _results(
            [
                [1, 2, "Market", "economy", 0.95],
                [2, 1, "Budget", "economy", 0.60],
                [3, 3, "Cricket", "sports", 0.20],
            ]
        )

    def test_equal_weights_combine_scores(self):
        combined = combine_rankings(
            self.tfidf,
            self.word2vec,
            alpha=0.5,
            top_k=3,
        )

        self.assertEqual(combined.iloc[0]["document_id"], 1)
        self.assertAlmostEqual(combined.iloc[0]["hybrid_score"], 0.75)

    def test_lower_alpha_can_favor_semantic_result(self):
        combined = combine_rankings(
            self.tfidf,
            self.word2vec,
            alpha=0.2,
            top_k=3,
        )

        self.assertEqual(combined.iloc[0]["document_id"], 2)

    def test_tfidf_fallback_when_word2vec_is_empty(self):
        empty = pd.DataFrame(columns=self.word2vec.columns)
        combined = combine_rankings(
            self.tfidf,
            empty,
            alpha=0.2,
            top_k=2,
        )

        self.assertEqual(combined["document_id"].tolist(), [1, 2])
        self.assertAlmostEqual(combined.iloc[0]["hybrid_score"], 0.90)
        self.assertAlmostEqual(combined.iloc[0]["word2vec_score"], 0.0)

    def test_word2vec_fallback_when_tfidf_is_empty(self):
        empty = pd.DataFrame(columns=self.tfidf.columns)
        combined = combine_rankings(
            empty,
            self.word2vec,
            alpha=0.8,
            top_k=1,
        )

        self.assertEqual(combined.iloc[0]["document_id"], 2)
        self.assertAlmostEqual(combined.iloc[0]["hybrid_score"], 0.95)

    def test_hybrid_searcher_runs_both_searchers(self):
        searcher = HybridSearcher(
            StaticSearcher(self.tfidf),
            StaticSearcher(self.word2vec),
            alpha=0.5,
        )

        results = searcher.search("economy query", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            list(results.columns),
            [
                "rank",
                "document_id",
                "title",
                "category",
                "hybrid_score",
                "tfidf_score",
                "word2vec_score",
            ],
        )

    def test_alpha_must_be_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            HybridSearcher(
                StaticSearcher(self.tfidf),
                StaticSearcher(self.word2vec),
                alpha=1.2,
            )

    def test_empty_query_returns_empty_result(self):
        searcher = HybridSearcher(
            StaticSearcher(self.tfidf),
            StaticSearcher(self.word2vec),
        )

        self.assertTrue(searcher.search("   ").empty)


if __name__ == "__main__":
    unittest.main()
