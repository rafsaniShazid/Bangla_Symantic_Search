import pandas as pd
import pytest

from src.tfidf_search import TFIDFSearcher


@pytest.fixture
def documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"document_id": "a", "title": "বাংলাদেশ বাজেট", "category": "দেশ", "text": "বাংলাদেশ বাজেট ঘোষণা"},
            {"document_id": "b", "title": "ঢাকার বৃষ্টি", "category": "দেশ", "text": "ঢাকায় আজ ভারী বৃষ্টি"},
            {"document_id": "c", "title": "ক্রিকেট ম্যাচ", "category": "খেলা", "text": "বাংলাদেশ ক্রিকেট ম্যাচ জিতেছে"},
        ]
    )


def test_search_ranks_matching_document_first(documents: pd.DataFrame) -> None:
    searcher = TFIDFSearcher().fit(documents)

    results = searcher.search("বাংলাদেশ বাজেট", top_k=2)

    assert results["document_id"].tolist() == ["a", "c"]
    assert results.iloc[0]["rank"] == 1
    assert results.iloc[0]["similarity_score"] > 0


def test_search_handles_empty_query(documents: pd.DataFrame) -> None:
    searcher = TFIDFSearcher().fit(documents)

    results = searcher.search("  ")

    assert results.empty
    assert results.columns.tolist() == [
        "rank", "document_id", "title", "category", "similarity_score"
    ]


def test_fit_requires_contract_columns() -> None:
    with pytest.raises(ValueError, match="document_id, title, and category"):
        TFIDFSearcher().fit(pd.DataFrame({"text": ["সংবাদ"]}))


def test_search_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="Call fit"):
        TFIDFSearcher().search("সংবাদ")
