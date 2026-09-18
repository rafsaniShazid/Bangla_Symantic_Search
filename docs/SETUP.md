# Development Setup

This project uses Python 3.12 for a stable Windows environment with Gensim,
scikit-learn, Streamlit, and the other NLP dependencies.

## Create the environment

From the project root:

```powershell
conda env create -f environment.yml
conda activate nlp312
python --version
```

Expected version:

```text
Python 3.12.x
```

## Run the current tests

```powershell
python -m pytest -q
```

## Start the Streamlit app later

```powershell
python -m streamlit run app.py
```

## Returning to the project

```powershell
cd "<project-root>"
conda activate nlp312
```

To leave the environment:

```powershell
conda deactivate
```

The environment file is for reproducible local setup. Real datasets, trained
models, and environment folders should not be committed to Git.
