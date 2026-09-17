import numpy as np
import pandas as pd
import pytest

from src.word2vec_search import (
    Word2VecSearcher,
    get_document_vector,
    get_query_vector,
)


class FakeVectors:
    vector_size = 2
    key_to_index = {"বাংলাদেশ": 0, "বাজেট": 1, "খেলা": 2}
    values = {
        "বাংলাদেশ": np.array([1.0, 0.0]),
        "বাজেট": np.array([0.9, 0.1]),
        "খেলা": np.array([0.0, 1.0]),
    }

    def __getitem__(self, word: str) -> np.ndarray:
        return self.values[word]


class FakeModel:
    wv = FakeVectors()


@pytest.fixture
def documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"document_id": "a", "title": "বাজেট", "category": "দেশ", "text": "বাংলাদেশ বাজেট"},
            {"document_id": "b", "title": "খেলা", "category": "খেলা", "text": "খেলা"},
        ]
    )


def test_mean_pooling_ignores_oov_tokens() -> None:
    vector = get_document_vector(["বাংলাদেশ", "অজানা"], FakeModel())

    assert np.allclose(vector, [1.0, 0.0])
    assert get_query_vector(["অজানা"], FakeModel()) is None


def test_word2vec_search_ranks_by_cosine_similarity(documents: pd.DataFrame) -> None:
    searcher = Word2VecSearcher().fit(documents, FakeModel())

    results = searcher.search("বাংলাদেশ", top_k=2)

    assert results["document_id"].tolist() == ["a", "b"]
    assert results.iloc[0]["similarity_score"] > 0.99


def test_all_oov_query_fails_gracefully(documents: pd.DataFrame) -> None:
    searcher = Word2VecSearcher().fit(documents, FakeModel())

    results = searcher.search("সম্পূর্ণ অজানা শব্দ")

    assert results.empty
    assert "No query words" in searcher.last_message
