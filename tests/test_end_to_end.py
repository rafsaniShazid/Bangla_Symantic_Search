"""Small end-to-end check across the full retrieval pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.data_loader import load_dataset
from src.dataset_pipeline import prepare_dataset
from src.evaluation_runner import (
    evaluate_methods,
    summarize_evaluation,
)
from src.system import (
    METHOD_CUSTOM_W2V,
    METHOD_HYBRID_CUSTOM,
    METHOD_TFIDF,
    SemanticSearchSystem,
)


def build_test_vectors():
    vectors = KeyedVectors(vector_size=4)

    words = [
        "বাংলাদেশ",
        "অর্থনীতি",
        "বাজেট",
        "বাজার",
        "ক্রিকেট",
        "দল",
        "ম্যাচ",
        "প্রযুক্তি",
        "গবেষণা",
    ]

    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    vectors.add_vectors(words, matrix)
    return vectors


class TestEndToEndPipeline(unittest.TestCase):
    def test_dataset_search_hybrid_and_evaluation_work_together(self):
        with TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            raw_path = root / "news.csv"
            processed_path = root / "processed.csv"

            pd.DataFrame(
                [
                    {
                        "id": "n1",
                        "title": "বাংলাদেশের বাজেট",
                        "content": "বাংলাদেশ অর্থনীতি বাজেট বাজার সংবাদ",
                        "category": "economy",
                    },
                    {
                        "id": "n2",
                        "title": "ক্রিকেট দলের জয়",
                        "content": "বাংলাদেশ ক্রিকেট দল ম্যাচ জিতেছে",
                        "category": "sports",
                    },
                    {
                        "id": "n3",
                        "title": "প্রযুক্তি গবেষণা",
                        "content": "প্রযুক্তি গবেষণা নতুন সংবাদ",
                        "category": "technology",
                    },
                ]
            ).to_csv(
                raw_path,
                index=False,
                encoding="utf-8-sig",
            )

            processed, statistics = prepare_dataset(
                data_path=raw_path,
                output_path=processed_path,
            )

            self.assertEqual(statistics.article_count, 3)
            self.assertTrue(processed_path.is_file())
            self.assertIn("processed_text", processed.columns)

            documents, _ = load_dataset(raw_path)
            system = SemanticSearchSystem(
                documents=documents,
                custom_model=build_test_vectors(),
                alpha=0.5,
            )

            self.assertEqual(
                system.available_methods(),
                [
                    METHOD_TFIDF,
                    METHOD_CUSTOM_W2V,
                    METHOD_HYBRID_CUSTOM,
                ],
            )

            tfidf_results = system.search(
                "অর্থনীতি বাজেট",
                method=METHOD_TFIDF,
                top_k=1,
            )
            self.assertEqual(
                tfidf_results.iloc[0]["document_id"],
                "n1",
            )

            word2vec_results = system.search(
                "ক্রিকেট ম্যাচ",
                method=METHOD_CUSTOM_W2V,
                top_k=1,
            )
            self.assertEqual(
                word2vec_results.iloc[0]["document_id"],
                "n2",
            )

            hybrid_results = system.search(
                "প্রযুক্তি গবেষণা",
                method=METHOD_HYBRID_CUSTOM,
                top_k=1,
                alpha=0.5,
            )
            self.assertEqual(
                hybrid_results.iloc[0]["document_id"],
                "n3",
            )
            self.assertIn(
                "hybrid_score",
                hybrid_results.columns,
            )

            queries = pd.DataFrame(
                [
                    {
                        "query_id": "Q1",
                        "query": "অর্থনীতি বাজেট",
                        "query_type": "lexical",
                        "relevant_ids": ["n1"],
                    },
                    {
                        "query_id": "Q2",
                        "query": "ক্রিকেট ম্যাচ",
                        "query_type": "semantic",
                        "relevant_ids": ["n2"],
                    },
                ]
            )

            details = evaluate_methods(
                system=system,
                queries=queries,
                methods=[
                    METHOD_TFIDF,
                    METHOD_CUSTOM_W2V,
                    METHOD_HYBRID_CUSTOM,
                ],
                k=1,
                alpha=0.5,
            )

            summary = summarize_evaluation(
                details,
                k=1,
            )

            self.assertEqual(len(summary), 3)
            self.assertTrue(
                (summary["precision@1"] >= 0.0).all()
            )
            self.assertTrue(
                (summary["recall@1"] >= 0.0).all()
            )
            self.assertTrue(
                (summary["mrr"] >= 0.0).all()
            )


if __name__ == "__main__":
    unittest.main()
