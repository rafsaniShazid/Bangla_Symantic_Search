"""Load pretrained Word2Vec vectors from common Gensim-compatible formats."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_SUFFIXES = {".model", ".kv", ".vec", ".txt", ".bin"}


def load_pretrained_word2vec(model_path: str | Path):
    """Load a pretrained Word2Vec/KeyedVectors model from disk.

    Supported formats:
    - Gensim Word2Vec model: .model
    - Gensim KeyedVectors model: .kv
    - word2vec text vectors: .vec or .txt
    - word2vec binary vectors: .bin
    """

    from gensim.models import KeyedVectors, Word2Vec

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained Word2Vec model was not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported pretrained model format: {suffix or '<no extension>'}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".model":
        return Word2Vec.load(str(path))

    if suffix == ".kv":
        return KeyedVectors.load(str(path), mmap="r")

    if suffix in {".vec", ".txt"}:
        return KeyedVectors.load_word2vec_format(str(path), binary=False)

    return KeyedVectors.load_word2vec_format(str(path), binary=True)


def get_keyed_vectors(model):
    """Return the keyed-vector view from a Word2Vec or KeyedVectors object."""

    return getattr(model, "wv", model)


def model_summary(model) -> dict[str, int]:
    """Return simple model information for UI/debugging."""

    keyed_vectors = get_keyed_vectors(model)
    vocabulary = getattr(keyed_vectors, "key_to_index", {})
    return {
        "vocabulary_size": len(vocabulary),
        "vector_size": int(keyed_vectors.vector_size),
    }


def most_similar_words(model, word: str, topn: int = 5) -> list[tuple[str, float]]:
    """Return nearest words for a vocabulary term."""

    if topn < 1:
        raise ValueError("topn must be at least 1.")

    keyed_vectors = get_keyed_vectors(model)
    if word not in keyed_vectors.key_to_index:
        return []

    return [
        (candidate, float(score))
        for candidate, score in keyed_vectors.most_similar(word, topn=topn)
    ]
