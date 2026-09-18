"""Streamlit interface for Bangla news semantic search."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.comparison import compare_methods
from src.embedding_explorer import (
    CUSTOM_EMBEDDING,
    PRETRAINED_EMBEDDING,
    available_embedding_models,
    embedding_neighbors,
    embedding_summary,
)
from src.evaluation_data import (
    count_completed_judgements,
    load_evaluation_queries,
)
from src.evaluation_runner import (
    evaluate_methods,
    summarize_by_query_type,
    summarize_evaluation,
)
from src.explanations import explain_method
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

EMBEDDING_LABELS = {
    CUSTOM_EMBEDDING: "Custom Word2Vec",
    PRETRAINED_EMBEDDING: "Pretrained Word2Vec",
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


def render_search_explanation(system, query, method, results, alpha):
    """Render compact, model-specific explanation details."""

    details = explain_method(system, query, method)

    with st.expander("How this search was interpreted"):
        tfidf_details = details["tfidf"]
        word2vec_details = details["word2vec"]

        if tfidf_details is not None:
            st.markdown("**TF-IDF query processing**")
            st.write(
                "Processed tokens:",
                ", ".join(tfidf_details["tokens"]) or "None",
            )
            st.write(
                "Matched vocabulary tokens:",
                ", ".join(tfidf_details["matched_tokens"]) or "None",
            )
            st.write(
                "Unmatched tokens:",
                ", ".join(tfidf_details["unmatched_tokens"]) or "None",
            )

            if tfidf_details["active_features"]:
                feature_table = pd.DataFrame(
                    tfidf_details["active_features"]
                )
                feature_table["weight"] = feature_table["weight"].map(
                    lambda value: round(float(value), 4)
                )
                st.dataframe(
                    feature_table,
                    use_container_width=True,
                    hide_index=True,
                )

        if word2vec_details is not None:
            st.markdown("**Word2Vec vocabulary coverage**")
            st.write(
                "Known words:",
                ", ".join(word2vec_details["known_words"]) or "None",
            )
            st.write(
                "OOV words:",
                ", ".join(word2vec_details["oov_words"]) or "None",
            )
            st.progress(
                float(word2vec_details["coverage"]),
                text=(
                    "Vocabulary coverage: "
                    f"{word2vec_details['coverage'] * 100:.1f}%"
                ),
            )

        if method in HYBRID_METHODS and not results.empty:
            st.markdown("**Hybrid score breakdown**")
            st.caption(
                f"Hybrid = {alpha:.2f} × TF-IDF + "
                f"{1.0 - alpha:.2f} × Word2Vec"
            )
            breakdown = results[
                [
                    "rank",
                    "document_id",
                    "tfidf_score",
                    "word2vec_score",
                    "hybrid_score",
                ]
            ].copy()
            for column in [
                "tfidf_score",
                "word2vec_score",
                "hybrid_score",
            ]:
                breakdown[column] = breakdown[column].map(
                    lambda value: round(float(value), 4)
                )
            st.dataframe(
                breakdown,
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
embedding_models = available_embedding_models(system)

with st.sidebar:
    st.caption(f"Documents loaded: {len(system.documents)}")
    st.caption(f"Methods ready: {len(available_methods)}")

search_tab, compare_tab, embedding_tab, evaluation_tab = st.tabs(
    ["Search", "Compare Models", "Embeddings", "Evaluation"]
)

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

                try:
                    render_search_explanation(
                        system=system,
                        query=query,
                        method=selected_method,
                        results=results,
                        alpha=alpha,
                    )
                except Exception as error:
                    st.caption(f"Explanation unavailable: {error}")

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

with embedding_tab:
    st.subheader("Word2Vec Embedding Explorer")
    st.caption(
        "Inspect the nearest words learned by a loaded Word2Vec model."
    )

    if not embedding_models:
        st.info(
            "No Word2Vec model is currently loaded. Train the custom model "
            "or provide a pretrained model path from the sidebar."
        )
    else:
        selected_embedding = st.selectbox(
            "Embedding model",
            options=list(embedding_models.keys()),
            format_func=lambda name: EMBEDDING_LABELS.get(name, name),
            key="embedding_model",
        )

        selected_model = embedding_models[selected_embedding]
        summary = embedding_summary(selected_model)

        metric_1, metric_2 = st.columns(2)
        metric_1.metric(
            "Vocabulary size",
            f"{summary['vocabulary_size']:,}",
        )
        metric_2.metric(
            "Vector dimension",
            summary["vector_size"],
        )

        embedding_word = st.text_input(
            "Bangla word",
            placeholder="উদাহরণ: অর্থনীতি",
            key="embedding_word",
        )

        neighbor_count = st.slider(
            "Number of similar words",
            min_value=1,
            max_value=20,
            value=10,
            key="embedding_topn",
        )

        if st.button(
            "Find similar words",
            type="primary",
            use_container_width=True,
            key="embedding_button",
        ):
            if not embedding_word.strip():
                st.warning("Enter a Bangla word first.")
            else:
                neighbors = embedding_neighbors(
                    selected_model,
                    embedding_word,
                    topn=neighbor_count,
                )

                if neighbors.empty:
                    st.warning(
                        "The word was not found in this Word2Vec vocabulary."
                    )
                else:
                    neighbors["similarity"] = neighbors["similarity"].map(
                        lambda value: round(float(value), 4)
                    )
                    st.dataframe(
                        neighbors,
                        use_container_width=True,
                        hide_index=True,
                    )

with evaluation_tab:
    st.subheader("Retrieval Evaluation")
    st.caption(
        "Evaluate judged queries using Precision@K, Recall@K, and MRR."
    )

    try:
        evaluation_queries = load_evaluation_queries()
    except Exception as error:
        st.error(f"Could not load evaluation queries: {error}")
    else:
        total_queries = len(evaluation_queries)
        judged_queries = count_completed_judgements(evaluation_queries)

        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Evaluation queries", total_queries)
        metric_2.metric("Judged queries", judged_queries)

        status_table = evaluation_queries[
            [
                "query_id",
                "query",
                "query_type",
                "relevant_document_ids",
            ]
        ].copy()
        status_table["status"] = status_table["relevant_document_ids"].map(
            lambda value: "Judged" if str(value).strip() else "Pending"
        )
        st.dataframe(
            status_table,
            use_container_width=True,
            hide_index=True,
        )

        if judged_queries == 0:
            st.info(
                "The 24 evaluation queries are prepared, but relevance "
                "judgements are still blank. Add real document IDs to "
                "`data/evaluation/qrels.csv` before reporting evaluation scores."
            )
        else:
            evaluation_k = st.slider(
                "Evaluation K",
                min_value=1,
                max_value=10,
                value=5,
                key="evaluation_k",
            )

            evaluation_alpha = st.slider(
                "Hybrid TF-IDF weight (alpha)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="evaluation_alpha",
            )

            evaluation_methods = st.multiselect(
                "Methods to evaluate",
                options=available_methods,
                default=available_methods,
                format_func=lambda method: METHOD_LABELS.get(method, method),
                key="evaluation_methods",
            )

            if st.button(
                "Run evaluation",
                type="primary",
                use_container_width=True,
                key="evaluation_button",
            ):
                if not evaluation_methods:
                    st.warning("Select at least one retrieval method.")
                else:
                    try:
                        details = evaluate_methods(
                            system=system,
                            queries=evaluation_queries,
                            methods=evaluation_methods,
                            k=evaluation_k,
                            alpha=evaluation_alpha,
                        )
                        summary = summarize_evaluation(
                            details,
                            k=evaluation_k,
                        )
                        by_type = summarize_by_query_type(
                            details,
                            k=evaluation_k,
                        )
                    except Exception as error:
                        st.error(f"Evaluation failed: {error}")
                    else:
                        precision_column = f"precision@{evaluation_k}"
                        recall_column = f"recall@{evaluation_k}"

                        display_summary = summary.copy()
                        display_summary["method"] = display_summary["method"].map(
                            lambda method: METHOD_LABELS.get(method, method)
                        )
                        for column in [
                            precision_column,
                            recall_column,
                            "mrr",
                        ]:
                            display_summary[column] = display_summary[column].map(
                                lambda value: round(float(value), 4)
                            )

                        st.markdown("#### Overall model performance")
                        st.dataframe(
                            display_summary,
                            use_container_width=True,
                            hide_index=True,
                        )

                        chart_data = display_summary.set_index("method")[
                            [
                                precision_column,
                                recall_column,
                                "mrr",
                            ]
                        ]
                        st.bar_chart(chart_data)

                        st.markdown("#### Performance by query type")
                        display_by_type = by_type.copy()
                        display_by_type["method"] = display_by_type["method"].map(
                            lambda method: METHOD_LABELS.get(method, method)
                        )
                        for column in [
                            precision_column,
                            recall_column,
                            "mrr",
                        ]:
                            display_by_type[column] = display_by_type[column].map(
                                lambda value: round(float(value), 4)
                            )

                        st.dataframe(
                            display_by_type,
                            use_container_width=True,
                            hide_index=True,
                        )

                        with st.expander("Per-query evaluation details"):
                            display_details = details.copy()
                            display_details["method"] = display_details["method"].map(
                                lambda method: METHOD_LABELS.get(method, method)
                            )
                            for column in [
                                precision_column,
                                recall_column,
                                "reciprocal_rank",
                            ]:
                                display_details[column] = display_details[column].map(
                                    lambda value: round(float(value), 4)
                                )
                            st.dataframe(
                                display_details,
                                use_container_width=True,
                                hide_index=True,
                            )
