"""Simple Streamlit interface for Bangla news search."""

import streamlit as st

import config
from src.system import build_search_system


METHODS = {"TF-IDF": "tfidf", "Word2Vec": "word2vec", "Hybrid": "hybrid"}


@st.cache_resource
def load_system():
    return build_search_system()


def show_results(results):
    """Display the ranked results with one common score column."""

    if results.empty:
        st.info("No results. The query may contain only unknown words.")
        return
    table = results.rename(
        columns={"similarity_score": "score", "hybrid_score": "score"}
    ).copy()
    score_columns = [column for column in table if column.endswith("score")]
    table[score_columns] = table[score_columns].round(4)
    st.dataframe(table, hide_index=True)


st.set_page_config(page_title="Bangla News Search", page_icon="🔎")
st.title("Bangla News Semantic Search")

if not config.DEFAULT_DATA_PATH.is_file():
    st.error(f"News CSV missing: {config.DEFAULT_DATA_PATH}")
    st.stop()
if not config.CUSTOM_W2V_PATH.is_file():
    st.error("Custom Word2Vec model missing. Run `python -m src.train_word2vec` first.")
    st.stop()

try:
    system = load_system()
except Exception as error:
    st.error(f"Could not load the search system: {error}")
    st.stop()

search_tab, compare_tab = st.tabs(["Search", "Compare"])

with search_tab:
    query = st.text_input("Bangla query", key="search_query")
    label = st.selectbox("Retrieval method", list(METHODS), key="search_method")
    top_k = st.slider("Top-K", 1, 20, 10, key="search_k")
    alpha = 0.5
    if label == "Hybrid":
        alpha = st.slider(
            "TF-IDF weight (alpha)", 0.0, 1.0, 0.5, 0.05, key="search_alpha"
        )
    if st.button("Search", key="search_button"):
        if query.strip():
            show_results(system.search(query, METHODS[label], top_k, alpha))
        else:
            st.warning("Enter a Bangla query.")

with compare_tab:
    query = st.text_input("Bangla query", key="compare_query")
    top_k = st.slider("Top-K", 1, 20, 5, key="compare_k")
    alpha = st.slider(
        "TF-IDF weight (alpha)", 0.0, 1.0, 0.5, 0.05, key="compare_alpha"
    )
    if st.button("Compare", key="compare_button"):
        if query.strip():
            for label, method in METHODS.items():
                st.subheader(label)
                show_results(system.search(query, method, top_k, alpha))
        else:
            st.warning("Enter a Bangla query.")
