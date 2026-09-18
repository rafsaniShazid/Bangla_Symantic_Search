"""Tests for retrieval evaluation metrics."""

import unittest

from src.evaluation import (
    evaluate_query,
    mean_evaluation,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestEvaluationMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        score = precision_at_k(
            retrieved_ids=[1, 2, 3, 4],
            relevant_ids=[2, 4, 8],
            k=4,
        )
        self.assertAlmostEqual(score, 0.5)

    def test_precision_uses_k_as_denominator(self):
        score = precision_at_k(
            retrieved_ids=[1, 2],
            relevant_ids=[1, 2],
            k=4,
        )
        self.assertAlmostEqual(score, 0.5)

    def test_recall_at_k(self):
        score = recall_at_k(
            retrieved_ids=[1, 2, 3, 4],
            relevant_ids=[2, 4, 8, 9],
            k=4,
        )
        self.assertAlmostEqual(score, 0.5)

    def test_recall_is_zero_when_no_relevant_documents_are_defined(self):
        self.assertEqual(
            recall_at_k([1, 2], [], k=2),
            0.0,
        )

    def test_reciprocal_rank_uses_first_relevant_result(self):
        score = reciprocal_rank(
            retrieved_ids=[5, 8, 2, 9],
            relevant_ids=[2, 9],
        )
        self.assertAlmostEqual(score, 1 / 3)

    def test_reciprocal_rank_is_zero_when_no_hit_exists(self):
        self.assertEqual(
            reciprocal_rank([1, 2, 3], [9]),
            0.0,
        )

    def test_document_ids_are_compared_as_strings(self):
        score = precision_at_k(
            retrieved_ids=[1, 2],
            relevant_ids=["1"],
            k=2,
        )
        self.assertAlmostEqual(score, 0.5)

    def test_evaluate_query_returns_all_metrics(self):
        metrics = evaluate_query(
            retrieved_ids=[2, 5, 7],
            relevant_ids=[5, 8],
            k=3,
        )

        self.assertAlmostEqual(metrics["precision@3"], 1 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 0.5)
        self.assertAlmostEqual(metrics["reciprocal_rank"], 0.5)

    def test_mean_evaluation_returns_precision_recall_and_mrr(self):
        metrics = mean_evaluation(
            [
                ([1, 2, 3], [1]),
                ([4, 5, 6], [5]),
            ],
            k=3,
        )

        self.assertAlmostEqual(metrics["precision@3"], 1 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.75)

    def test_empty_evaluation_set_returns_zeroes(self):
        metrics = mean_evaluation([], k=5)

        self.assertEqual(
            metrics,
            {
                "precision@5": 0.0,
                "recall@5": 0.0,
                "mrr": 0.0,
            },
        )

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            precision_at_k([1], [1], k=0)


if __name__ == "__main__":
    unittest.main()
