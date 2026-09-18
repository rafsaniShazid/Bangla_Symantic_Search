"""Tests for the retrieval evaluation query structure."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.evaluation_data import (
    count_completed_judgements,
    load_evaluation_queries,
    parse_relevant_document_ids,
)


class TestEvaluationData(unittest.TestCase):
    def test_project_query_file_has_balanced_query_types(self):
        queries = load_evaluation_queries(
            Path("data") / "evaluation" / "qrels.csv"
        )

        self.assertEqual(len(queries), 24)
        self.assertEqual(
            queries["query_type"].value_counts().to_dict(),
            {"lexical": 8, "semantic": 8, "mixed": 8},
        )

    def test_relevance_ids_are_intentionally_empty_initially(self):
        queries = load_evaluation_queries(
            Path("data") / "evaluation" / "qrels.csv"
        )

        self.assertEqual(count_completed_judgements(queries), 0)

    def test_parse_pipe_separated_relevant_ids(self):
        self.assertEqual(
            parse_relevant_document_ids("n1|n7|n12"),
            ["n1", "n7", "n12"],
        )

    def test_loader_rejects_invalid_query_type(self):
        with TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "bad.csv"
            pd.DataFrame(
                [
                    {
                        "query_id": "Q1",
                        "query": "বাংলাদেশ",
                        "query_type": "unknown",
                        "relevant_document_ids": "n1",
                    }
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            with self.assertRaises(ValueError):
                load_evaluation_queries(path)


if __name__ == "__main__":
    unittest.main()
