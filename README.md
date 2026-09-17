# Bangla News Semantic Search

This repository will compare explainable Bangla news retrieval using TF-IDF, pretrained Word2Vec, and custom-trained Word2Vec. The project is being built in phases so the shared dataset contract is verified before retrieval and evaluation are added.

## Phase 1: Dataset Loader

The current implementation provides the project structure and a reusable CSV loader in `src/data_loader.py`. It does not yet perform Bangla linguistic preprocessing or search.

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

`text` is the unprocessed combination of title and content. Unicode normalization, tokenization, punctuation handling, and stopword removal belong to the shared preprocessing phase.

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

Run the Phase 1 tests:

```powershell
pytest tests/test_data_loader.py
```

## Planned phases

1. Dataset contract and loader
2. Shared Bangla preprocessing
3. TF-IDF baseline and search tests
4. Custom and pretrained Word2Vec search
5. Evaluation, comparison, error analysis, visualization, and application interface
