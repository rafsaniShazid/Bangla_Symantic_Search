"""Streamlit search interface for the Bangla news retrieval system."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
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
            display[column] = display[column].map(lambda value: round(float(value), 4))

    return display


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
    st.header("Search settings")

    selected_method = st.selectbox(
        "Retrieval method",
        options=available_methods,
        format_func=lambda method: METHOD_LABELS.get(method, method),
    )

    top_k = st.slider(
        "Top K results",
        min_value=1,
        max_value=20,
        value=min(10, max(1, len(system.documents))),
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
        )

    st.caption(f"Documents loaded: {len(system.documents)}")
    st.caption(f"Methods ready: {len(available_methods)}")

query = st.text_input(
    "Search query",
    placeholder="উদাহরণ: বাংলাদেশের অর্থনৈতিক অবস্থা",
)

search_clicked = st.button(
    "Search",
    type="primary",
    use_container_width=True,
)

if search_clicked:
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
            st.subheader(
                f"Results — {METHOD_LABELS.get(selected_method, selected_method)}"
            )

            if results.empty:
                st.warning(
                    "No result was returned. For Word2Vec, this can happen when "
                    "all query words are outside the model vocabulary."
                )
            else:
                st.dataframe(
                    format_results(results),
                    use_container_width=True,
                    hide_index=True,
                )
