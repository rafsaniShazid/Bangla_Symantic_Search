"""Central configuration for the Bangla news search project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

MODELS_DIR = PROJECT_ROOT / "models"
CUSTOM_W2V_DIR = MODELS_DIR / "word2vec_custom"
PRETRAINED_W2V_DIR = MODELS_DIR / "word2vec_pretrained"
PRETRAINED_W2V_PATH = PRETRAINED_W2V_DIR / "model.bin"

DEFAULT_DATA_PATH = RAW_DATA_DIR / "news.csv"

# Logical field names map to source CSV column names here.
DEFAULT_COLUMN_MAPPING = {
    "title": "title",
    "content": "content",
    "category": "category",
}
SOURCE_ID_COLUMN = "id"

# Common corpus-specific alternatives are used only when the configured name
# is absent; explicit column_mapping values always take precedence.
COLUMN_ALIASES = {
    "content": ("content", "text", "description"),
}

RANDOM_SEED = 42

REQUIRED_LOGICAL_COLUMNS = ("title", "content", "category")
OUTPUT_COLUMNS = ("document_id", "title", "content", "category", "text")

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 1.0
DEFAULT_TOP_K = 10

W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 1
W2V_SG = 1
W2V_EPOCHS = 10
CUSTOM_W2V_PATH = MODELS_DIR / "word2vec_custom" / "bangla_news.model"
