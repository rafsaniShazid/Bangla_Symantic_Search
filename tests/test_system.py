"""Tests for the three-method search dispatcher."""

import numpy as np
import pandas as pd
import pytest

from src.system import SemanticSearchSystem


class FakeVectors:
    vector_size = 2
    values = {
        "বাংলাদেশ": [1.0, 0.0],
        "অর্থনীতি": [1.0, 0.0],
        "বাজেট": [1.0, 0.0],
        "ক্রিকেট": [0.0, 1.0],
        "ম্যাচ": [0.0, 1.0],
        "প্রযুক্তি": [-1.0, 0.0],
    }
    key_to_index = {word: index for index, word in enumerate(values)}

    def __getitem__(self, word):
        return np.array(self.values[word])


class FakeModel:
    wv = FakeVectors()


@pytest.fixture
def system():
    documents = pd.DataFrame(
        [
            {"document_id": "n1", "title": "বাজেট", "category": "economy", "text": "বাংলাদেশ অর্থনীতি বাজেট"},
            {"document_id": "n2", "title": "ক্রিকেট", "category": "sports", "text": "ক্রিকেট ম্যাচ"},
            {"document_id": "n3", "title": "প্রযুক্তি", "category": "tech", "text": "প্রযুক্তি সংবাদ"},
        ]
    )
    return SemanticSearchSystem(documents, FakeModel())


def test_only_three_methods_are_available(system):
    assert system.available_methods() == ["tfidf", "word2vec", "hybrid"]


@pytest.mark.parametrize("method", ["tfidf", "word2vec", "hybrid"])
def test_search_dispatch_ranks_budget_first(system, method):
    results = system.search("অর্থনীতি বাজেট", method=method, top_k=1)
    assert results.iloc[0]["document_id"] == "n1"


def test_hybrid_uses_requested_alpha(system):
    results = system.search("অর্থনীতি বাজেট", method="hybrid", top_k=1, alpha=0.3)
    row = results.iloc[0]
    assert row["hybrid_score"] == pytest.approx(
        0.3 * row["tfidf_score"] + 0.7 * row["word2vec_score"]
    )


def test_unknown_method_is_rejected(system):
    with pytest.raises(ValueError, match="Unknown retrieval method"):
        system.search("বাংলাদেশ", method="unknown")
