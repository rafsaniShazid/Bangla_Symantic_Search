# Bangla News Semantic Search

Search Bangla news with three methods: **TF-IDF**, **custom Word2Vec**, and a **hybrid** of their scores. The Streamlit app has Search and Compare tabs.

## Pipeline

```text
News CSV → Unicode normalization, cleaning, tokenization, stopword removal
         ├→ TF-IDF (unigrams and bigrams) → cosine similarity ─┐
         └→ custom Skip-gram Word2Vec → mean pooling → cosine similarity ─┤
                                           weighted hybrid score ←┘
                                                      ↓
                                            Top-K and comparison
                                                      ↓
                                   Precision@K, Recall@K, MRR when judged
```

`src/preprocessing.py` is shared by training and both searchers. The hybrid is **score fusion**, not a trained model:

```text
hybrid_score = alpha * tfidf_score + (1 - alpha) * word2vec_score
```

The default `alpha` is `0.5`. If a query has no known words for one method, hybrid uses the available method's score.

## Setup and use

Use the Python 3.12 environment from `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate nlp312
```

Place the news corpus at `data/raw/news.csv`. It needs `title`, `category`, and either `content` or `text`; `id` is optional. The loader removes empty and duplicate articles and returns `document_id`, `title`, `content`, `category`, and combined `text`. The raw CSV and trained model are ignored by Git.

Train the custom model and start the app:

```powershell
python -m src.train_word2vec
python -m streamlit run app.py
```

The model is saved to `models/word2vec_custom/bangla_news.model`. Search takes one query, one method, Top-K, and an alpha slider for Hybrid. Compare runs the same query through all three methods.

Current settings in `config.py`: TF-IDF `ngram_range=(1, 2)`, Word2Vec `vector_size=100`, `window=5`, `min_count=2`, `sg=1` (Skip-gram), `epochs=10`, and default Top-K `10`.

## Evaluation and tests

`data/evaluation/qrels.csv` contains prepared queries. Add real `document_id` values to `relevant_document_ids`, separated by `|`. Its judgements are currently blank, so this project does not claim measured model performance. The backend skips unjudged queries:

```python
from src.evaluation import evaluate_system, load_evaluation_queries
from src.system import build_search_system

results = evaluate_system(build_search_system(), load_evaluation_queries(), k=5)
print(results)  # Empty until real relevance judgements are added.
```

Run the complete suite with `python -m pytest -q`.
