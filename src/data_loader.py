"""Load and validate the shared Bangla news document contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

import config
from src.preprocessing import preprocess_many


class DatasetError(ValueError):
    """Raised when a dataset cannot satisfy the project data contract."""


@dataclass(frozen=True)
class DatasetStatistics:
    """Summary statistics for a cleaned news dataset."""

    article_count: int
    category_count: int
    articles_per_category: dict[str, int]
    average_article_length: float


def _clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _deduplication_key(title: str, content: str) -> str:
    normalized = f"{title} {content}".casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _validate_source_ids(values: pd.Series) -> pd.Series:
    cleaned = values.map(_clean_value)
    if cleaned.eq("").any():
        raise DatasetError("The source ID column contains empty values.")
    if cleaned.duplicated().any():
        raise DatasetError("The source ID column must contain unique values.")
    return cleaned


def _resolve_mapping(column_mapping: Mapping[str, str] | None) -> dict[str, str]:
    mapping = dict(config.DEFAULT_COLUMN_MAPPING)
    if column_mapping:
        mapping.update(column_mapping)
    missing_logical = set(config.REQUIRED_LOGICAL_COLUMNS) - set(mapping)
    if missing_logical:
        missing = ", ".join(sorted(missing_logical))
        raise DatasetError(f"Column mapping is missing logical fields: {missing}.")
    return mapping


def _resolve_source_columns(
    mapping: Mapping[str, str], available_columns: pd.Index
) -> dict[str, str]:
    resolved = dict(mapping)
    for logical_name, source_column in mapping.items():
        if source_column in available_columns:
            continue
        aliases = config.COLUMN_ALIASES.get(logical_name, ())
        replacement = next(
            (alias for alias in aliases if alias in available_columns), None
        )
        if replacement is not None:
            resolved[logical_name] = replacement
    return resolved


def load_dataset(
    data_path: str | Path | None = None,
    column_mapping: Mapping[str, str] | None = None,
    source_id_column: str | None = config.SOURCE_ID_COLUMN,
) -> tuple[pd.DataFrame, DatasetStatistics]:
    """Load, clean, and summarize a news CSV.

    The returned frame always contains ``document_id``, ``title``, ``content``,
    ``category``, and ``text``. Linguistic preprocessing is intentionally left
    to ``src.preprocessing`` so all retrieval methods can share that pipeline.
    """

    path = Path(data_path) if data_path is not None else config.DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    raw = pd.read_csv(path)
    mapping = _resolve_source_columns(
        _resolve_mapping(column_mapping), raw.columns
    )
    missing_columns = [
        source_column
        for source_column in mapping.values()
        if source_column not in raw.columns
    ]
    if missing_columns:
        missing = ", ".join(sorted(set(missing_columns)))
        raise DatasetError(f"Dataset is missing required columns: {missing}")

    selected = pd.DataFrame(
        {
            logical_name: raw[source_column]
            for logical_name, source_column in mapping.items()
        }
    )
    for column in ("title", "content", "category"):
        selected[column] = selected[column].map(_clean_value)

    selected = selected.loc[
        selected["title"].ne("") & selected["content"].ne("")
    ].copy()
    selected["category"] = selected["category"].replace("", "Uncategorized")

    selected["_deduplication_key"] = [
        _deduplication_key(title, content)
        for title, content in zip(selected["title"], selected["content"])
    ]
    selected = selected.drop_duplicates("_deduplication_key", keep="first")

    if source_id_column and source_id_column in raw.columns:
        source_ids = _validate_source_ids(raw.loc[selected.index, source_id_column])
        selected["document_id"] = source_ids.to_numpy()
    else:
        selected["document_id"] = range(1, len(selected) + 1)

    selected["text"] = selected["title"] + " " + selected["content"]
    selected = selected[["document_id", "title", "content", "category", "text"]]
    selected = selected.reset_index(drop=True)

    statistics = summarize_dataset(selected)
    return selected, statistics


def summarize_dataset(documents: pd.DataFrame) -> DatasetStatistics:
    """Calculate deterministic statistics for a loader-contract DataFrame."""

    required = set(config.OUTPUT_COLUMNS)
    missing = required - set(documents.columns)
    if missing:
        raise DatasetError(
            "Cannot summarize documents missing columns: "
            + ", ".join(sorted(missing))
        )

    lengths = documents["text"].map(lambda text: len(str(text).split()))
    distribution = (
        documents["category"].value_counts(sort=False).sort_index().astype(int).to_dict()
    )
    return DatasetStatistics(
        article_count=len(documents),
        category_count=len(distribution),
        articles_per_category=distribution,
        average_article_length=float(lengths.mean()) if len(lengths) else 0.0,
    )


DEFAULT_PROCESSED_PATH = config.PROCESSED_DATA_DIR / "news_processed.csv"


def add_processed_text(documents: pd.DataFrame) -> pd.DataFrame:
    """Add the shared processed-text column without changing the input frame."""

    required = set(config.OUTPUT_COLUMNS)
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(
            "Documents are missing required columns: "
            + ", ".join(sorted(missing))
        )

    processed = documents.copy()
    processed["processed_text"] = preprocess_many(
        processed["text"].fillna("").astype(str).tolist()
    )
    return processed


def prepare_dataset(
    data_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_PROCESSED_PATH,
) -> tuple[pd.DataFrame, DatasetStatistics]:
    """Load, preprocess, and save a news corpus as a processed CSV."""

    documents, statistics = load_dataset(data_path)
    processed = add_processed_text(documents)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination, index=False, encoding="utf-8-sig")
    return processed, statistics
