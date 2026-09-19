"""Train a tiny custom model and run the complete search flow."""

import pandas as pd

from src.evaluation import evaluate_system, load_evaluation_queries
from src.system import build_search_system
from src.train_word2vec import train_custom_word2vec


def test_training_search_and_judged_evaluation(tmp_path):
    data_path = tmp_path / "news.csv"
    model_path = tmp_path / "bangla_news.model"
    qrels_path = tmp_path / "qrels.csv"

    pd.DataFrame(
        [
            {"id": "n1", "title": "বাজেট", "content": "অর্থনীতি বাজেট অর্থনীতি বাজেট", "category": "economy"},
            {"id": "n2", "title": "ক্রিকেট", "content": "ক্রিকেট ম্যাচ ক্রিকেট ম্যাচ", "category": "sports"},
            {"id": "n3", "title": "প্রযুক্তি", "content": "প্রযুক্তি গবেষণা প্রযুক্তি গবেষণা", "category": "tech"},
        ]
    ).to_csv(data_path, index=False, encoding="utf-8-sig")

    model = train_custom_word2vec(data_path, model_path)
    assert model_path.is_file()
    assert model.sg == 1
    assert model.wv.vector_size == 100

    system = build_search_system(data_path, model_path)
    for method in system.available_methods():
        results = system.search("অর্থনীতি বাজেট", method=method, top_k=1)
        assert results.iloc[0]["document_id"] == "n1"

    pd.DataFrame(
        [
            {"query_id": "Q1", "query": "অর্থনীতি বাজেট", "query_type": "lexical", "relevant_document_ids": "n1"},
            {"query_id": "Q2", "query": "ক্রিকেট ম্যাচ", "query_type": "semantic", "relevant_document_ids": ""},
        ]
    ).to_csv(qrels_path, index=False, encoding="utf-8-sig")
    summary = evaluate_system(system, load_evaluation_queries(qrels_path), k=1)
    assert summary["method"].tolist() == ["tfidf", "word2vec", "hybrid"]
    assert summary["queries"].tolist() == [1, 1, 1]
    assert summary["precision@1"].tolist() == [1.0, 1.0, 1.0]
