"""Tests for the Bangla-aware TF-IDF adapter."""

import unittest

import pandas as pd

from src.tfidf_search import BanglaTFIDFSearcher


class TestBanglaTFIDFSearcher(unittest.TestCase):
    def setUp(self):
        self.documents = pd.DataFrame(
            {
                "document_id": [1, 2, 3],
                "title": [
                    "বাংলাদেশের অর্থনীতি",
                    "ক্রিকেট দলের জয়",
                    "প্রযুক্তি সংবাদ",
                ],
                "category": ["economy", "sports", "technology"],
                "text": [
                    "বাংলাদেশের অর্থনীতি এবং নতুন বাজেট নিয়ে আলোচনা",
                    "বাংলাদেশ ক্রিকেট দল ম্যাচ জিতেছে",
                    "কৃত্রিম বুদ্ধিমত্তা ও প্রযুক্তি নিয়ে নতুন গবেষণা",
                ],
            }
        )

    def test_vocabulary_keeps_complete_bangla_words(self):
        searcher = BanglaTFIDFSearcher().fit(self.documents)
        vocabulary = searcher.vectorizer.vocabulary_

        self.assertIn("অর্থনীতি", vocabulary)
        self.assertIn("ক্রিকেট", vocabulary)
        self.assertIn("প্রযুক্তি", vocabulary)

    def test_search_ranks_relevant_economy_document_first(self):
        searcher = BanglaTFIDFSearcher().fit(self.documents)
        results = searcher.search("নতুন বাজেট অর্থনীতি", top_k=2)

        self.assertEqual(results.iloc[0]["document_id"], 1)
        self.assertGreater(results.iloc[0]["similarity_score"], 0.0)

    def test_query_uses_same_stopword_handling_as_documents(self):
        searcher = BanglaTFIDFSearcher().fit(self.documents)

        direct = searcher.search("অর্থনীতি বাজেট", top_k=3)
        with_stopwords = searcher.search("এই অর্থনীতি এবং বাজেট", top_k=3)

        self.assertEqual(
            direct["document_id"].tolist(),
            with_stopwords["document_id"].tolist(),
        )

    def test_dataframe_metadata_is_preserved(self):
        searcher = BanglaTFIDFSearcher().fit(self.documents)
        results = searcher.search("ক্রিকেট", top_k=1)

        self.assertEqual(results.iloc[0]["title"], "ক্রিকেট দলের জয়")
        self.assertEqual(results.iloc[0]["category"], "sports")


if __name__ == "__main__":
    unittest.main()
