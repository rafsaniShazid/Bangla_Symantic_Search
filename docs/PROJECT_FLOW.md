# Bangla News Semantic Search - Project Flow

## Goal

Build a Bangla news search system that compares lexical and semantic retrieval while keeping the pipeline easy to test and explain.

## Core pipeline

```text
Bangla news dataset
        |
        v
Dataset loader
        |
        v
Bangla preprocessing
        |
        +-------------------+
        |                   |
        v                   v
     TF-IDF              Word2Vec
        |             custom / pretrained
        |                   |
        +---------+---------+
                  |
                  v
             Hybrid search
                  |
                  v
              Top-K results
                  |
        +---------+---------+
        |                   |
        v                   v
   Evaluation          Streamlit UI
```

## Current repository

The existing project already contains the initial repository structure, dataset loader, TF-IDF search, Word2Vec search, custom Word2Vec training code, and their focused tests.

These existing modules stay as the starting point:

```text
src/data_loader.py
src/tfidf_search.py
src/word2vec_search.py
src/train_word2vec.py
```

## Remaining implementation

The remaining work will be added in small, testable steps:

1. Shared Bangla preprocessing
2. Processed dataset pipeline
3. Bangla-aware TF-IDF integration
4. Pretrained Word2Vec integration
5. Hybrid retrieval
6. Precision@K, Recall@K, and MRR evaluation
7. Evaluation query set
8. Retrieval-system integration
9. Streamlit search interface
10. Model comparison view
11. Word embedding explorer
12. Search explanation details
13. Evaluation dashboard
14. End-to-end checks and final documentation

## Git workflow

- `nasif` keeps the original implementation history.
- `main` is the stable integration branch and starts from the current `nasif` history.
- Remaining feature work will be developed on a separate Saif feature branch after the basic `main` setup is complete.
- Each logical change should be tested before it is committed.
- Large datasets and trained model files should stay outside Git unless there is a specific reason to version them.
