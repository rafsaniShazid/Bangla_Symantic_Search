"""Tests for the shared Bangla preprocessing pipeline."""

import unittest

from src.preprocessing import (
    clean_text,
    normalize_unicode,
    preprocess_many,
    preprocess_text,
    tokenize_bangla,
)


class TestBanglaPreprocessing(unittest.TestCase):
    def test_bangla_words_are_not_split_into_letters(self):
        tokens = tokenize_bangla(
            "বাংলাদেশ অর্থনীতি ক্রিকেট",
            remove_stopwords=False,
        )
        self.assertEqual(tokens, ["বাংলাদেশ", "অর্থনীতি", "ক্রিকেট"])

    def test_clean_text_removes_url_html_and_punctuation(self):
        text = "<b>বাংলাদেশ!</b> https://example.com অর্থনীতি?"
        cleaned = clean_text(text)

        self.assertEqual(cleaned, "বাংলাদেশ অর্থনীতি")

    def test_stopwords_can_be_removed(self):
        processed = preprocess_text("এই দেশের অর্থনীতি এবং বাজেট")
        tokens = processed.split()

        self.assertNotIn("এই", tokens)
        self.assertNotIn("এবং", tokens)
        self.assertIn("অর্থনীতি", tokens)
        self.assertIn("বাজেট", tokens)

    def test_stopword_removal_can_be_disabled(self):
        tokens = tokenize_bangla(
            "এই দেশের অর্থনীতি",
            remove_stopwords=False,
        )

        self.assertIn("এই", tokens)

    def test_latin_text_and_numbers_are_kept(self):
        cleaned = clean_text("AI 2026: বাংলাদেশ")
        self.assertEqual(cleaned, "AI 2026 বাংলাদেশ")

    def test_preprocess_many_uses_the_same_pipeline(self):
        result = preprocess_many(
            ["বাংলাদেশ এবং অর্থনীতি", "ক্রিকেট ও খেলা"]
        )

        self.assertEqual(result, ["বাংলাদেশ অর্থনীতি", "ক্রিকেট খেলা"])

    def test_unicode_normalization_returns_a_string(self):
        normalized = normalize_unicode("বাংলাদেশ")
        self.assertEqual(normalized, "বাংলাদেশ")


if __name__ == "__main__":
    unittest.main()
