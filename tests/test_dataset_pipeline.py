"""Tests for the processed dataset pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.dataset_pipeline import add_processed_text, prepare_dataset


class TestDatasetPipeline(unittest.TestCase):
    def test_add_processed_text_keeps_document_contract(self):
        source = pd.DataFrame(
            {
                "document_id": [1],
                "title": ["অর্থনীতির খবর"],
                "content": ["এই বাজেট এবং অর্থনীতি নিয়ে আলোচনা।"],
                "category": ["economy"],
                "text": ["অর্থনীতির খবর এই বাজেট এবং অর্থনীতি নিয়ে আলোচনা।"],
            }
        )

        result = add_processed_text(source)

        self.assertEqual(len(result), 1)
        self.assertIn("processed_text", result.columns)
        self.assertIn("অর্থনীতি", result.loc[0, "processed_text"])
        self.assertNotIn("এবং", result.loc[0, "processed_text"])
        self.assertNotIn("processed_text", source.columns)

    def test_prepare_dataset_saves_processed_csv(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_path = root / "news.csv"
            output_path = root / "processed" / "news_processed.csv"

            pd.DataFrame(
                {
                    "id": ["n1", "n2"],
                    "title": ["বাংলাদেশের বাজেট", "ক্রিকেট সংবাদ"],
                    "content": [
                        "<p>এই বাজেট এবং অর্থনীতি নিয়ে খবর।</p>",
                        "বাংলাদেশ ক্রিকেট দল জিতেছে https://example.com",
                    ],
                    "category": ["economy", "sports"],
                }
            ).to_csv(raw_path, index=False, encoding="utf-8-sig")

            processed, statistics = prepare_dataset(raw_path, output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(statistics.article_count, 2)
            self.assertEqual(processed["document_id"].tolist(), ["n1", "n2"])
            self.assertIn("processed_text", processed.columns)
            self.assertNotIn("<p>", processed.loc[0, "processed_text"])
            self.assertNotIn("https", processed.loc[1, "processed_text"])

            saved = pd.read_csv(output_path)
            self.assertEqual(len(saved), 2)
            self.assertIn("processed_text", saved.columns)


if __name__ == "__main__":
    unittest.main()
