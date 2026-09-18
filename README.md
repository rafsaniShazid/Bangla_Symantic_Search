# Bangla News Semantic Search

This repository will compare explainable Bangla news retrieval using TF-IDF, pretrained Word2Vec, and custom-trained Word2Vec. The project is being built in phases so the shared dataset contract is verified before retrieval and evaluation are added.

## Phase 1: Dataset Loader

The current implementation provides the project structure, a reusable CSV loader in `src/data_loader.py`, and explainable TF-IDF and mean-pooled Word2Vec search modules. Bangla linguistic preprocessing remains a shared integration step.

### Expected dataset

Place the real dataset at:

```text
data/raw/news.csv
```

The default logical fields are:

| Logical field | Default CSV column | Required |
| --- | --- | --- |
| `title` | `title` | Yes |
| `content` | `content` | Yes |
| `category` | `category` | Yes |
| source ID | `id` | No |

The loader supports renamed source columns through the `column_mapping` argument. It returns this stable contract for every document:

```text
document_id, title, content, category, text
```

`text` is the unprocessed combination of title and content. The downloaded corpus uses `text` instead of `content`; the loader recognizes this through the configured content alias. Unicode normalization, tokenization, punctuation handling, and stopword removal belong to the shared preprocessing phase.

### Document IDs

When an `id` column exists, its non-empty unique values are preserved. Otherwise, the loader assigns 1-based IDs after removing invalid and duplicate rows. These IDs are the canonical keys for future evaluation annotations.

### Load data

```python
from src.data_loader import load_dataset

documents, statistics = load_dataset()
print(statistics)
print(documents.head())
```

The loader removes rows with an empty title or content, removes duplicate title/content pairs while retaining the first row, and reports article count, category distribution, and average whitespace-token length.

No real dataset or evaluation result is fabricated by this project. Any test fixture used later will be explicitly labeled demo data.

## Development

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Run the focused tests:

```powershell
pytest tests/test_data_loader.py tests/test_tfidf_search.py tests/test_word2vec_search.py
```

## Planned phases

1. Dataset contract and loader
2. Shared Bangla preprocessing
3. TF-IDF baseline and search tests
4. Custom and pretrained Word2Vec search
5. Evaluation, comparison, error analysis, visualization, and application interface

## Retrieval modules

`src/tfidf_search.py` provides `TFIDFSearcher`, which fits a scikit-learn TF-IDF matrix and ranks the shared document contract with cosine similarity. `src/word2vec_search.py` provides mean-pooled Word2Vec vectors, OOV-safe query handling, and the same ranked result columns.

Train the custom model with:

```powershell
python -m src.train_word2vec
```

The model is saved to `models/word2vec_custom/bangla_news.model`. A pretrained model is not downloaded automatically; its path will be configured explicitly when the application integration is added.

## Search application

CLI search:

```powershell
python app.py --query "বাংলাদেশের বাজেট ঘোষণা" --method tfidf --top-k 5
```

Start the Streamlit interface after installing the project requirements:

```powershell
streamlit run app.py
```

The interface supports TF-IDF, custom Word2Vec, and pretrained Word2Vec selections. Word2Vec options require the corresponding model file; no pretrained model is downloaded automatically.
