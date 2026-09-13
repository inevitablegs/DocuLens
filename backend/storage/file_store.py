"""
DocuLens AI — Local file storage manager
"""
import hashlib
import shutil
from pathlib import Path

from backend.config import DOCUMENTS_DIR


def compute_file_hash(file_bytes: bytes) -> str:
    """SHA-256 hash of file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_document_dir(doc_id: str) -> Path:
    """Return (and create) the storage directory for a document."""
    d = DOCUMENTS_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_uploaded_file(doc_id: str, filename: str, file_bytes: bytes) -> Path:
    """Save the original uploaded file and return its path."""
    doc_dir = get_document_dir(doc_id)
    ext = Path(filename).suffix.lower()
    dest = doc_dir / f"original{ext}"
    dest.write_bytes(file_bytes)
    return dest


def save_page_image(doc_id: str, page_number: int, image_bytes: bytes, fmt: str = "png") -> Path:
    """Save a rendered page image and return its path."""
    doc_dir = get_document_dir(doc_id)
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    dest = pages_dir / f"page_{page_number:03d}.{fmt}"
    dest.write_bytes(image_bytes)
    return dest


def get_original_path(doc_id: str) -> Path | None:
    """Find the original file for a document."""
    doc_dir = DOCUMENTS_DIR / doc_id
    if not doc_dir.exists():
        return None
    for f in doc_dir.iterdir():
        if f.stem == "original":
            return f
    return None


def get_page_image_path(doc_id: str, page_number: int) -> Path | None:
    """Get a page image path."""
    pages_dir = DOCUMENTS_DIR / doc_id / "pages"
    if not pages_dir.exists():
        return None
    for fmt in ("png", "jpg", "jpeg"):
        p = pages_dir / f"page_{page_number:03d}.{fmt}"
        if p.exists():
            return p
    return None


def delete_document_files(doc_id: str):
    """Remove all files for a document."""
    doc_dir = DOCUMENTS_DIR / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
