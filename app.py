"""CLI and Streamlit interface for Bangla news search."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from src.data_loader import load_dataset
from src.tfidf_search import TFIDFSearcher
from src.word2vec_search import Word2VecSearcher, load_word2vec_model

METHOD_LABELS = {
    "tfidf": "TF-IDF",
    "custom_w2v": "Custom Word2Vec",
    "pretrained_w2v": "Pretrained Word2Vec",
}


def build_searcher(method: str, data_path: str | Path | None = None):
    """Load the corpus and construct the requested fitted searcher."""

    if method not in METHOD_LABELS:
        valid_methods = ", ".join(METHOD_LABELS)
        raise ValueError(f"Unknown method '{method}'. Choose one of: {valid_methods}")

    documents, _ = load_dataset(data_path)
    if method == "tfidf":
        return TFIDFSearcher().fit(documents)

    model_path = (
        config.CUSTOM_W2V_PATH
        if method == "custom_w2v"
        else config.PRETRAINED_W2V_PATH
    )
    model = load_word2vec_model(model_path)
    return Word2VecSearcher().fit(documents, model)


def run_cli(query: str, method: str, top_k: int, data_path: str | Path | None = None) -> int:
    """Run one search from the command line and print ranked results."""

    try:
        searcher = build_searcher(method, data_path)
        results = searcher.search(query, top_k=top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    if results.empty:
        message = getattr(searcher, "last_message", "No matching documents found.")
        print(message or "No matching documents found.")
        return 0

    print(f"Method: {METHOD_LABELS[method]}")
    print(f"Query: {query}")
    for _, result in results.iterrows():
        print(
            f"{int(result['rank'])}. {result['title']} "
            f"[{result['category']}] "
            f"score={result['similarity_score']:.4f} "
            f"(document_id={result['document_id']})"
        )
    return 0


def run_streamlit(data_path: str | Path | None = None) -> None:
    """Render the focused Streamlit search interface."""

    import streamlit as st

    st.set_page_config(page_title="Bangla News Semantic Search", layout="wide")
    st.title("Bangla News Semantic Search")
    st.caption("Search the loaded Bangla news corpus with explainable retrieval methods.")

    @st.cache_data
    def load_documents(path: str | None):
        documents, statistics = load_dataset(path)
        return documents, statistics

    documents, statistics = load_documents(str(data_path) if data_path else None)
    st.sidebar.metric("Articles", statistics.article_count)
    st.sidebar.metric("Categories", statistics.category_count)
    method = st.sidebar.selectbox(
        "Retrieval method",
        options=list(METHOD_LABELS),
        format_func=METHOD_LABELS.get,
    )
    top_k = st.sidebar.selectbox("Top results", options=[1, 5, 10], index=2)
    query = st.text_input("Search query", placeholder="বাংলাদেশের নতুন বাজেট ঘোষণা")

    if not query.strip():
        st.info("Enter a Bangla query to search the corpus.")
        return

    try:
        @st.cache_resource(show_spinner="Loading retrieval model...")
        def create_searcher(selected_method: str):
            if selected_method == "tfidf":
                return TFIDFSearcher().fit(documents)
            model_path = (
                config.CUSTOM_W2V_PATH
                if selected_method == "custom_w2v"
                else config.PRETRAINED_W2V_PATH
            )
            return Word2VecSearcher().fit(documents, load_word2vec_model(model_path))

        searcher = create_searcher(method)
        results = searcher.search(query, top_k=top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(str(error))
        return

    if results.empty:
        st.warning(getattr(searcher, "last_message", "No matching documents found."))
        return

    st.subheader(f"{len(results)} result(s)")
    for _, result in results.iterrows():
        document_id = result["document_id"]
        article = documents.loc[documents["document_id"] == document_id]
        snippet = article.iloc[0]["content"] if not article.empty else ""
        st.markdown(f"### {int(result['rank'])}. {result['title']}")
        st.write(f"**Category:** {result['category']}  |  **Similarity:** {result['similarity_score']:.4f}")
        st.write(str(snippet)[:500] + ("..." if len(str(snippet)) > 500 else ""))
        st.divider()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Bangla query for one CLI search")
    parser.add_argument(
        "--method",
        choices=list(METHOD_LABELS),
        default="tfidf",
    )
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--data", type=Path, default=None)
    return parser.parse_known_args()[0]


def main() -> int:
    arguments = parse_args()
    if arguments.query is not None:
        return run_cli(arguments.query, arguments.method, arguments.top_k, arguments.data)
    run_streamlit(arguments.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
