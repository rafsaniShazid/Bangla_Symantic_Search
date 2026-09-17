from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DatasetError, load_dataset


def write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "news.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_dataset_cleans_duplicates_and_reports_statistics(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        [
            {"id": "a-10", "title": "বাজেট ঘোষণা", "content": "নতুন বাজেট প্রকাশিত হয়েছে", "category": "রাজনীতি"},
            {"id": "duplicate", "title": " বাজেট ঘোষণা ", "content": "নতুন বাজেট প্রকাশিত হয়েছে", "category": "রাজনীতি"},
            {"id": "a-30", "title": "ঢাকার আবহাওয়া", "content": "আজ বৃষ্টি হতে পারে", "category": "দেশ"},
            {"id": "empty", "title": "", "content": "ফাঁকা শিরোনাম বাদ যাবে", "category": "দেশ"},
        ],
    )

    documents, statistics = load_dataset(path)

    assert documents["document_id"].tolist() == ["a-10", "a-30"]
    assert documents["text"].tolist() == [
        "বাজেট ঘোষণা নতুন বাজেট প্রকাশিত হয়েছে",
        "ঢাকার আবহাওয়া আজ বৃষ্টি হতে পারে",
    ]
    assert statistics.article_count == 2
    assert statistics.category_count == 2
    assert statistics.articles_per_category == {"দেশ": 1, "রাজনীতি": 1}
    assert statistics.average_article_length == 6.0


def test_generated_ids_are_assigned_after_cleaning(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        [
            {"title": "প্রথম", "content": "একটি সংবাদ", "category": "দেশ"},
            {"title": "", "content": "বাদ যাবে", "category": "দেশ"},
            {"title": "দ্বিতীয়", "content": "আরেকটি সংবাদ", "category": "খেলা"},
        ],
    )

    documents, _ = load_dataset(path)

    assert documents["document_id"].tolist() == [1, 2]


def test_custom_column_mapping_is_supported(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        [
            {"headline": "শিরোনাম", "body": "সংবাদের লেখা", "section": "দেশ"},
        ],
    )

    documents, _ = load_dataset(
        path,
        column_mapping={"title": "headline", "content": "body", "category": "section"},
        source_id_column=None,
    )

    assert documents.iloc[0].to_dict() == {
        "document_id": 1,
        "title": "শিরোনাম",
        "content": "সংবাদের লেখা",
        "category": "দেশ",
        "text": "শিরোনাম সংবাদের লেখা",
    }


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file was not found"):
        load_dataset(tmp_path / "missing.csv")


def test_missing_required_column_raises_clear_error(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        [{"title": "শিরোনাম", "content": "লেখা"}],
    )

    with pytest.raises(DatasetError, match="missing required columns: category"):
        load_dataset(path)


def test_duplicate_source_ids_raise_clear_error(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        [
            {"id": "same", "title": "এক", "content": "লেখা এক", "category": "দেশ"},
            {"id": "same", "title": "দুই", "content": "লেখা দুই", "category": "দেশ"},
        ],
    )

    with pytest.raises(DatasetError, match="source ID column must contain unique values"):
        load_dataset(path)
