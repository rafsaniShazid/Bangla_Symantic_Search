"""Train and save a custom Gensim Word2Vec model on the news corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from src.data_loader import load_dataset
from src.preprocessing import tokenize_bangla


def train_custom_word2vec(
    data_path: str | Path | None = None,
    output_path: str | Path = config.CUSTOM_W2V_PATH,
):
    """Train a reproducible custom Word2Vec model and save it to disk."""

    from gensim.models import Word2Vec

    documents, statistics = load_dataset(data_path)
    sentences = [tokenize_bangla(text) for text in documents["text"]]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        raise ValueError("The corpus produced no tokenized documents.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=config.W2V_VECTOR_SIZE,
        window=config.W2V_WINDOW,
        min_count=config.W2V_MIN_COUNT,
        workers=config.W2V_WORKERS,
        sg=config.W2V_SG,
        seed=config.RANDOM_SEED,
        epochs=config.W2V_EPOCHS,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))

    print(f"Articles used: {statistics.article_count}")
    print(f"Vocabulary size: {len(model.wv.key_to_index)}")
    print(f"Embedding dimension: {model.wv.vector_size}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=config.CUSTOM_W2V_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    train_custom_word2vec(arguments.data, arguments.output)


if __name__ == "__main__":
    main()
