# Project Usage Guide

This project compares lexical TF-IDF retrieval with Word2Vec-based semantic
retrieval for Bangla news articles.

## 1. Activate the environment

```powershell
conda activate nlp312
```

## 2. Add the news dataset

Place the real corpus at:

```text
data/raw/news.csv
```

The existing data loader expects the project news fields such as title,
content, and category. An id column is used as the source document ID when it
is available.

## 3. Train the custom Word2Vec model

```powershell
python -m src.train_word2vec --data data/raw/news.csv --output models/word2vec_custom/bangla_news.model
```

The custom model is trained only on the project corpus.

## 4. Optional pretrained Word2Vec model

A genuine external pretrained model can be supplied separately. Supported
formats are:

```text
.model
.kv
.vec
.txt
.bin
```

Keep large pretrained model files outside normal Git history. In the Streamlit
sidebar, enter the local path to the model you are actually using.

## 5. Run all tests

```powershell
python -m pytest -q
```

## 6. Start the application

```powershell
python -m streamlit run app.py
```

The application provides:

- Search
- Compare Models
- Embeddings
- Evaluation

Depending on which model files are available, the system can expose:

- TF-IDF
- Custom Word2Vec
- Pretrained Word2Vec
- Hybrid TF-IDF + Custom Word2Vec
- Hybrid TF-IDF + Pretrained Word2Vec

## 7. Evaluation workflow

Evaluation queries are stored in:

```text
data/evaluation/qrels.csv
```

Each row has:

```text
query_id
query
query_type
relevant_document_ids
```

`query_type` is one of:

```text
lexical
semantic
mixed
```

Relevant document IDs must be judged against the real corpus. Multiple
relevant IDs are separated with `|`, for example:

```text
n12|n47|n103
```

Do not report Precision@K, Recall@K, or MRR until the relevance judgements are
filled using real document IDs from the dataset.

After the judgements are completed, open the Evaluation tab to compare the
retrieval methods.
