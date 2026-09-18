"""Streamlit interface for Bangla news semantic search."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from src.comparison import compare_methods
from src.system import (
    METHOD_CUSTOM_W2V,
    METHOD_HYBRID_CUSTOM,
    METHOD_HYBRID_PRETRAINED,
    METHOD_PRETRAINED_W2V,
    METHOD_TFIDF,
    build_search_system,
)


METHOD_LABELS = {
    METHOD_TFIDF: "TF-IDF (Lexical)",
    METHOD_CUSTOM_W2V: "Custom Word2Vec (Semantic)",
    METHOD_PRETRAINED_W2V: "Pretrained Word2Vec (Semantic)",
    METHOD_HYBRID_CUSTOM: "Hybrid: TF-IDF + Custom Word2Vec",
    METHOD_HYBRID_PRETRAINED: "Hybrid: TF-IDF + Pretrained Word2Vec",
}

HYBRID_METHODS = {
    METHOD_HYBRID_CUSTOM,
    METHOD_HYBRID_PRETRAINED,
}


@st.cache_resource(show_spinner=False)
def load_search_system(
    data_path: str,
    custom_model_path: str,
    pretrained_model_path: str,
):
    """Build and cache the retrieval system for the selected files."""

    custom_path = (
        custom_model_path
        if custom_model_path and Path(custom_model_path).is_file()
        else None
    )
    pretrained_path = (
        pretrained_model_path
        if pretrained_model_path and Path(pretrained_model_path).is_file()
        else None
    )

    return build_search_system(
        data_path=data_path,
        custom_model_path=custom_path,
        pretrained_model_path=pretrained_path,
    )


def format_results(results):
    """Return a display copy with readable score names."""

    display = results.copy()

    rename_map = {
        "similarity_score": "score",
        "hybrid_score": "hybrid",
        "tfidf_score": "tfidf",
        "word2vec_score": "word2vec",
    }
    display = display.rename(columns=rename_map)

    for column in ["score", "hybrid", "tfidf", "word2vec"]:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: round(float(value), 4)
            )

    return display


def render_comparison_grid(comparison, methods):
    """Show up to three retrieval methods side-by-side per row."""

    for start in range(0, len(methods), 3):
        group = methods[start : start + 3]
        columns = st.columns(len(group))

        for column, method in zip(columns, group):
            with column:
                st.markdown(f"#### {METHOD_LABELS.get(method, method)}")
                method_results = comparison[
                    comparison["method"] == method
                ][
                    ["rank", "document_id", "title", "category", "score"]
                ].copy()

                if method_results.empty:
                    st.info("No result returned.")
                else:
                    method_results["score"] = method_results["score"].map(
                        lambda value: round(float(value), 4)
                    )
                    st.dataframe(
                        method_results,
                        use_container_width=True,
                        hide_index=True,
                    )


st.set_page_config(
    page_title="Bangla News Semantic Search",
    page_icon="🔎",
    layout="wide",
)

st.title("Bangla News Semantic Search")
st.caption(
    "Compare lexical TF-IDF retrieval with Word2Vec-based semantic retrieval."
)

with st.sidebar:
    st.header("Project files")

    data_path = st.text_input(
        "News CSV",
        value=str(config.DEFAULT_DATA_PATH),
        help="CSV used by the existing project data loader.",
    )

    custom_model_path = st.text_input(
        "Custom Word2Vec model",
        value=str(config.CUSTOM_W2V_PATH),
        help="Optional. Leave the path as-is if the trained model exists there.",
    )

    pretrained_model_path = st.text_input(
        "Pretrained Word2Vec model",
        value="",
        placeholder="models/word2vec_pretrained/model.vec",
        help="Optional external pretrained model: .model, .kv, .vec, .txt, or .bin",
    )

    if st.button("Reload project files"):
        load_search_system.clear()
        st.rerun()

if not Path(data_path).is_file():
    st.info(
        "News dataset not found. Put the corpus at "
        f"`{data_path}` or choose another CSV path from the sidebar."
    )
    st.stop()

try:
    with st.spinner("Loading retrieval models..."):
        system = load_search_system(
            data_path,
            custom_model_path,
            pretrained_model_path,
        )
except Exception as error:
    st.error(f"Could not load the search system: {error}")
    st.stop()

available_methods = system.available_methods()

with st.sidebar:
    st.caption(f"Documents loaded: {len(system.documents)}")
    st.caption(f"Methods ready: {len(available_methods)}")

search_tab, compare_tab = st.tabs(["Search", "Compare Models"])

with search_tab:
    st.subheader("Search")

    selected_method = st.selectbox(
        "Retrieval method",
        options=available_methods,
        format_func=lambda method: METHOD_LABELS.get(method, method),
        key="search_method",
    )

    top_k = st.slider(
        "Top K results",
        min_value=1,
        max_value=20,
        value=min(10, max(1, len(system.documents))),
        key="search_top_k",
    )

    alpha = 0.5
    if selected_method in HYBRID_METHODS:
        alpha = st.slider(
            "TF-IDF weight (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Hybrid = alpha × TF-IDF + (1 - alpha) × Word2Vec",
            key="search_alpha",
        )

    query = st.text_input(
        "Search query",
        placeholder="উদাহরণ: বাংলাদেশের অর্থনৈতিক অবস্থা",
        key="search_query",
    )

    if st.button(
        "Search",
        type="primary",
        use_container_width=True,
        key="search_button",
    ):
        if not query.strip():
            st.warning("Enter a Bangla query first.")
        else:
            try:
                results = system.search(
                    query=query,
                    method=selected_method,
                    top_k=top_k,
                    alpha=alpha if selected_method in HYBRID_METHODS else None,
                )
            except Exception as error:
                st.error(f"Search failed: {error}")
            else:
                st.markdown(
                    f"#### {METHOD_LABELS.get(selected_method, selected_method)}"
                )

                if results.empty:
                    st.warning(
                        "No result was returned. For Word2Vec, this can happen "
                        "when all query words are outside the model vocabulary."
                    )
                else:
                    st.dataframe(
                        format_results(results),
                        use_container_width=True,
                        hide_index=True,
                    )

with compare_tab:
    st.subheader("Compare Models")
    st.caption(
        "Run the same Bangla query through every model that is currently loaded."
    )

    compare_query = st.text_input(
        "Comparison query",
        placeholder="উদাহরণ: দেশের অর্থনৈতিক অবস্থা",
        key="compare_query",
    )

    compare_top_k = st.slider(
        "Results per method",
        min_value=1,
        max_value=10,
        value=min(5, max(1, len(system.documents))),
        key="compare_top_k",
    )

    compare_alpha = st.slider(
        "Hybrid TF-IDF weight (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="compare_alpha",
    )

    selected_compare_methods = st.multiselect(
        "Methods to compare",
        options=available_methods,
        default=available_methods,
        format_func=lambda method: METHOD_LABELS.get(method, method),
    )

    if len(available_methods) == 1:
        st.info(
            "Only TF-IDF is currently loaded. Add a custom or pretrained "
            "Word2Vec model to compare lexical and semantic retrieval."
        )

    if st.button(
        "Compare",
        type="primary",
        use_container_width=True,
        key="compare_button",
    ):
        if not compare_query.strip():
            st.warning("Enter a Bangla comparison query first.")
        elif not selected_compare_methods:
            st.warning("Select at least one retrieval method.")
        else:
            try:
                comparison = compare_methods(
                    system=system,
                    query=compare_query,
                    methods=selected_compare_methods,
                    top_k=compare_top_k,
                    alpha=compare_alpha,
                )
            except Exception as error:
                st.error(f"Comparison failed: {error}")
            else:
                render_comparison_grid(
                    comparison,
                    selected_compare_methods,
                )
