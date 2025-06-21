import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
FEEDBACK_DIR = DATA_DIR / "feedback"
MODELS_DIR = DATA_DIR / "models"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K_DOCUMENTS = 5
MIN_SCORE_THRESHOLD = 0.3

FEEDBACK_FILE = FEEDBACK_DIR / "feedback.json"
MIN_FEEDBACK_FOR_TRAINING = 5

DATABASE_URL = f"sqlite:///{DATA_DIR}/rag_system.db"

WEB_HOST = "0.0.0.0"
WEB_PORT = 8000