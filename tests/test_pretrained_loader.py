"""Tests for pretrained Word2Vec model loading."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from src.word2vec_search import (
    load_word2vec_model,
    model_summary,
    most_similar_words,
)


class TestPretrainedWord2VecLoader(unittest.TestCase):
    def _build_vectors(self):
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

    def test_load_native_keyedvectors(self):
        with TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "tiny.kv"
            vectors = self._build_vectors()
            vectors.save(str(path))

            loaded = load_word2vec_model(path)

            self.assertEqual(loaded.vector_size, 3)
            self.assertIn("অর্থনীতি", loaded.key_to_index)

    def test_load_text_word2vec_format(self):
        with TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "tiny.vec"
            vectors = self._build_vectors()
            vectors.save_word2vec_format(str(path), binary=False)

            loaded = load_word2vec_model(path)

            self.assertEqual(loaded.vector_size, 3)
            self.assertIn("বাজেট", loaded.key_to_index)

    def test_load_binary_word2vec_format(self):
        with TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "tiny.bin"
            vectors = self._build_vectors()
            vectors.save_word2vec_format(str(path), binary=True)

            loaded = load_word2vec_model(path)

            self.assertEqual(loaded.vector_size, 3)
            self.assertIn("ক্রিকেট", loaded.key_to_index)

    def test_load_native_word2vec_model(self):
        with TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "tiny.model"
            model = Word2Vec(
                sentences=[
                    ["অর্থনীতি", "বাজেট"],
                    ["ক্রিকেট", "দল"],
                    ["অর্থনীতি", "বাজার"],
                ],
                vector_size=10,
                min_count=1,
                workers=1,
                seed=42,
                epochs=5,
            )
            model.save(str(path))

            loaded = load_word2vec_model(path)

            self.assertEqual(loaded.wv.vector_size, 10)
            self.assertIn("অর্থনীতি", loaded.wv.key_to_index)

    def test_summary_reports_size_and_dimensions(self):
        vectors = self._build_vectors()
        summary = model_summary(vectors)

        self.assertEqual(summary["vocabulary_size"], 3)
        self.assertEqual(summary["vector_size"], 3)

    def test_most_similar_returns_empty_for_oov_word(self):
        vectors = self._build_vectors()
        self.assertEqual(most_similar_words(vectors, "অজানা"), [])

    def test_rejects_unsupported_extension(self):
        with TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "tiny.xyz"
            path.write_text("not a model", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_word2vec_model(path)


if __name__ == "__main__":
    unittest.main()
