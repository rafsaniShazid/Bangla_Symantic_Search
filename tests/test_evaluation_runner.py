"""Tests for the evaluation runner and dashboard summaries."""

import unittest

import pandas as pd

from src.evaluation_runner import (
    completed_queries,
    evaluate_method,
    evaluate_methods,
    summarize_by_query_type,
    summarize_evaluation,
)


class FakeSystem:
    def available_methods(self):
        return ["tfidf", "custom_word2vec"]

    def search(self, query, method, top_k=5, alpha=None):
        if method == "tfidf":
            ids = {
                "budget query": ["n1", "n2", "n3"],
                "semantic query": ["n5", "n4", "n2"],
            }.get(query, [])
        else:
            ids = {
                "budget query": ["n2", "n1", "n3"],
                "semantic query": ["n4", "n5", "n2"],
            }.get(query, [])

        return pd.DataFrame(
            {
                "document_id": ids[:top_k],
            }
        )


class TestEvaluationRunner(unittest.TestCase):
    def setUp(self):
        self.queries = pd.DataFrame(
            [
                {
                    "query_id": "Q1",
                    "query": "budget query",
                    "query_type": "lexical",
                    "relevant_ids": ["n1", "n8"],
                },
                {
                    "query_id": "Q2",
                    "query": "semantic query",
                    "query_type": "semantic",
                    "relevant_ids": ["n4"],
                },
                {
                    "query_id": "Q3",
                    "query": "not judged",
                    "query_type": "mixed",
                    "relevant_ids": [],
                },
            ]
        )

    def test_completed_queries_filters_unjudged_rows(self):
        judged = completed_queries(self.queries)

        self.assertEqual(judged["query_id"].tolist(), ["Q1", "Q2"])

    def test_evaluate_method_uses_only_judged_queries(self):
        details = evaluate_method(
            FakeSystem(),
            self.queries,
            method="tfidf",
            k=2,
        )

        self.assertEqual(len(details), 2)
        self.assertEqual(details["query_id"].tolist(), ["Q1", "Q2"])
        self.assertIn("precision@2", details.columns)
        self.assertIn("recall@2", details.columns)

    def test_evaluate_methods_combines_method_results(self):
        details = evaluate_methods(
            FakeSystem(),
            self.queries,
            k=2,
        )

        self.assertEqual(
            details["method"].unique().tolist(),
            ["tfidf", "custom_word2vec"],
        )
        self.assertEqual(len(details), 4)

    def test_summary_returns_precision_recall_and_mrr(self):
        details = evaluate_methods(
            FakeSystem(),
            self.queries,
            methods=["tfidf"],
            k=2,
        )
        summary = summarize_evaluation(details, k=2)

        self.assertEqual(summary.iloc[0]["method"], "tfidf")
        self.assertEqual(summary.iloc[0]["queries"], 2)
        self.assertAlmostEqual(summary.iloc[0]["precision@2"], 0.5)
        self.assertAlmostEqual(summary.iloc[0]["recall@2"], 0.75)
        self.assertAlmostEqual(summary.iloc[0]["mrr"], 0.75)

    def test_query_type_summary_keeps_lexical_and_semantic_separate(self):
        details = evaluate_methods(
            FakeSystem(),
            self.queries,
            methods=["tfidf"],
            k=2,
        )
        summary = summarize_by_query_type(details, k=2)

        self.assertEqual(
            set(summary["query_type"]),
            {"lexical", "semantic"},
        )

    def test_no_judgements_returns_empty_evaluation(self):
        unjudged = self.queries.copy()
        unjudged["relevant_ids"] = [[], [], []]

        details = evaluate_methods(
            FakeSystem(),
            unjudged,
            k=2,
        )

        self.assertTrue(details.empty)

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_method(
                FakeSystem(),
                self.queries,
                method="tfidf",
                k=0,
            )


if __name__ == "__main__":
    unittest.main()
