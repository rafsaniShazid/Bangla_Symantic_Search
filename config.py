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

DEFAULT_DATA_PATH = RAW_DATA_DIR / "news.csv"

# Logical field names map to source CSV column names here.
DEFAULT_COLUMN_MAPPING = {
    "title": "title",
    "content": "content",
    "category": "category",
}
SOURCE_ID_COLUMN = "id"

RANDOM_SEED = 42

REQUIRED_LOGICAL_COLUMNS = ("title", "content", "category")
OUTPUT_COLUMNS = ("document_id", "title", "content", "category", "text")
