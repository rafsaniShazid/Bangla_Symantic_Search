"""Single integration layer for all retrieval methods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.data_loader import load_dataset
from src.hybrid_search import HybridSearcher
from src.preprocessing import tokenize_bangla
from src.tfidf_search import BanglaTFIDFSearcher
from src.word2vec_search import Word2VecSearcher, load_word2vec_model


METHOD_TFIDF = "tfidf"
METHOD_CUSTOM_W2V = "custom_word2vec"
METHOD_PRETRAINED_W2V = "pretrained_word2vec"
METHOD_HYBRID_CUSTOM = "hybrid_custom"
METHOD_HYBRID_PRETRAINED = "hybrid_pretrained"


class SemanticSearchSystem:
    """Fit and expose the project's lexical and semantic search methods."""

    def __init__(
        self,
        documents: pd.DataFrame,
        custom_model=None,
        pretrained_model=None,
        alpha: float = 0.5,
    ) -> None:
        if documents.empty:
            raise ValueError("The search system needs at least one document.")

        self.documents = documents.reset_index(drop=True).copy()
        self.alpha = float(alpha)

        self.tfidf = BanglaTFIDFSearcher().fit(self.documents)

        self.custom_word2vec = None
        self.pretrained_word2vec = None
        self.hybrid_custom = None
        self.hybrid_pretrained = None

        if custom_model is not None:
            self.custom_word2vec = self._build_word2vec_searcher(custom_model)
            self.hybrid_custom = HybridSearcher(
                self.tfidf,
                self.custom_word2vec,
                alpha=self.alpha,
            )

        if pretrained_model is not None:
            self.pretrained_word2vec = self._build_word2vec_searcher(
                pretrained_model
            )
            self.hybrid_pretrained = HybridSearcher(
                self.tfidf,
                self.pretrained_word2vec,
                alpha=self.alpha,
            )

    def _build_word2vec_searcher(self, model) -> Word2VecSearcher:
        searcher = Word2VecSearcher(tokenizer=tokenize_bangla)
        searcher.fit(self.documents, model, text_column="text")
        return searcher

    def available_methods(self) -> list[str]:
        """Return only the retrieval methods that are ready to use."""

        methods = [METHOD_TFIDF]

        if self.custom_word2vec is not None:
            methods.extend([METHOD_CUSTOM_W2V, METHOD_HYBRID_CUSTOM])

        if self.pretrained_word2vec is not None:
            methods.extend([METHOD_PRETRAINED_W2V, METHOD_HYBRID_PRETRAINED])

        return methods

    def search(
        self,
        query: str,
        method: str = METHOD_TFIDF,
        top_k: int = config.DEFAULT_TOP_K,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Search with one selected retrieval method."""

        if method == METHOD_TFIDF:
            return self.tfidf.search(query, top_k=top_k)

        if method == METHOD_CUSTOM_W2V:
            if self.custom_word2vec is None:
                raise RuntimeError("Custom Word2Vec model is not loaded.")
            return self.custom_word2vec.search(query, top_k=top_k)

        if method == METHOD_PRETRAINED_W2V:
            if self.pretrained_word2vec is None:
                raise RuntimeError("Pretrained Word2Vec model is not loaded.")
            return self.pretrained_word2vec.search(query, top_k=top_k)

        if method == METHOD_HYBRID_CUSTOM:
            if self.hybrid_custom is None:
                raise RuntimeError("Custom Word2Vec model is not loaded.")
            return self.hybrid_custom.search(
                query,
                top_k=top_k,
                alpha=alpha,
            )

        if method == METHOD_HYBRID_PRETRAINED:
            if self.hybrid_pretrained is None:
                raise RuntimeError("Pretrained Word2Vec model is not loaded.")
            return self.hybrid_pretrained.search(
                query,
                top_k=top_k,
                alpha=alpha,
            )

        raise ValueError(
            f"Unknown retrieval method: {method}. "
            f"Available methods: {', '.join(self.available_methods())}"
        )


def build_search_system(
    data_path: str | Path | None = None,
    custom_model_path: str | Path | None = None,
    pretrained_model_path: str | Path | None = None,
    alpha: float = 0.5,
) -> SemanticSearchSystem:
    """Load project files from disk and build one integrated search system."""

    documents, _ = load_dataset(data_path)

    custom_model = None
    if custom_model_path is not None:
        custom_model = load_word2vec_model(custom_model_path)

    pretrained_model = None
    if pretrained_model_path is not None:
        pretrained_model = load_word2vec_model(pretrained_model_path)

    return SemanticSearchSystem(
        documents=documents,
        custom_model=custom_model,
        pretrained_model=pretrained_model,
        alpha=alpha,
    )
