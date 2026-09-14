"""
DocuLens AI — Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PAGES_DIR = DATA_DIR / "pages"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "doculens.db"

# Ensure dirs exist
for d in [DOCUMENTS_DIR, PAGES_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Database ────────────────────────────────────────────
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# ─── Gemini ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAnKClHlLkbBvebJoLaieCH0xSvQQWjN8Y")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ─── Processing ──────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
CONFIDENCE_THRESHOLD = 0.85  # Below this → human review

# ─── Confidence Weights ──────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "ocr": 0.30,
    "extraction": 0.25,
    "schema": 0.20,
    "evidence": 0.15,
    "model": 0.10,
}
