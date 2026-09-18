"""Build the processed news dataset used by the search models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.data_loader import DatasetStatistics, load_dataset
from src.preprocessing import preprocess_many


DEFAULT_PROCESSED_PATH = config.PROCESSED_DATA_DIR / "news_processed.csv"


def add_processed_text(documents: pd.DataFrame) -> pd.DataFrame:
    """Add one shared processed-text column without changing the input frame."""

    required = {"document_id", "title", "content", "category", "text"}
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
    """Load the raw corpus, preprocess it, and save the processed CSV."""

    documents, statistics = load_dataset(data_path)
    processed = add_processed_text(documents)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination, index=False, encoding="utf-8-sig")

    return processed, statistics
