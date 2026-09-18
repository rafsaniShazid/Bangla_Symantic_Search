"""Bangla-aware text preprocessing shared by the retrieval models."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable


# A compact stopword list is enough for the lab project.
# It can be extended later after checking the final news corpus.
BANGLA_STOPWORDS = {
    "এ",
    "এই",
    "এবং",
    "ও",
    "আর",
    "যে",
    "যা",
    "যার",
    "যাদের",
    "যদি",
    "তবে",
    "করা",
    "করে",
    "করেছে",
    "করেন",
    "করতে",
    "হয়",
    "হয়",
    "হবে",
    "হলো",
    "হতে",
    "থেকে",
    "দিকে",
    "দিয়ে",
    "দিয়ে",
    "জন্য",
    "সঙ্গে",
    "উপর",
    "নিয়ে",
    "নিয়ে",
    "একটি",
    "এক",
    "কোনো",
    "কিছু",
    "অনেক",
    "মধ্যে",
    "পর",
    "আগে",
    "পরে",
    "এর",
    "কে",
    "কি",
    "কী",
    "না",
    "তো",
    "বা",
    "সহ",
    "প্রতি",
    "সব",
    "তাদের",
    "তার",
    "তিনি",
    "তারা",
    "আমরা",
    "আপনি",
    "আমি",
    "এটা",
    "সেটা",
    "ছিল",
    "ছিলেন",
    "আছে",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
NON_TOKEN_RE = re.compile(r"[^\u0980-\u09FFA-Za-z0-9]+")


def normalize_unicode(text: str) -> str:
    """Return text in canonical NFC Unicode form."""

    return unicodedata.normalize("NFC", str(text or ""))


def clean_text(text: str) -> str:
    """Remove markup, URLs, punctuation, and repeated whitespace."""

    cleaned = normalize_unicode(text)
    cleaned = html.unescape(cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = NON_TOKEN_RE.sub(" ", cleaned)
    return MULTISPACE_RE.sub(" ", cleaned).strip()


def tokenize_bangla(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize cleaned text without splitting Bangla characters incorrectly."""

    tokens = clean_text(text).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in BANGLA_STOPWORDS]
    return tokens


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """Return a whitespace-tokenized string for TF-IDF and Word2Vec."""

    return " ".join(tokenize_bangla(text, remove_stopwords=remove_stopwords))


def preprocess_many(
    texts: Iterable[str],
    remove_stopwords: bool = True,
) -> list[str]:
    """Preprocess a sequence of texts using the same rules."""

    return [
        preprocess_text(text, remove_stopwords=remove_stopwords)
        for text in texts
    ]
