"""
DocuLens AI — Document ingestion service
"""
import mimetypes
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.config import MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from backend.database.models import Document
from backend.storage.file_store import compute_file_hash, save_uploaded_file


class IngestionError(Exception):
    pass


async def validate_file(filename: str, file_bytes: bytes) -> tuple[str, str]:
    """Validate file type and size. Returns (extension, mime_type)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise IngestionError(f"File too large: {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB")

    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return ext, mime_type


async def check_duplicate(session: AsyncSession, file_hash: str) -> Document | None:
    """Check if a document with the same hash already exists."""
    result = await session.execute(
        select(Document).where(Document.file_hash == file_hash).order_by(Document.created_at.desc())
    )
    return result.scalars().first()


async def ingest_document(
    session: AsyncSession, filename: str, file_bytes: bytes
) -> tuple[Document, bool]:
    """
    Ingest a document: validate, hash, check duplicate, save.
    Returns (document, is_new). If is_new is False, it's a cache hit.
    """
    ext, mime_type = await validate_file(filename, file_bytes)
    file_hash = compute_file_hash(file_bytes)

    # Check for duplicate
    existing = await check_duplicate(session, file_hash)
    if existing and existing.status == "completed":
        return existing, False  # Cache hit!

    # Create new document record
    doc = Document(
        filename=filename,
        file_hash=file_hash,
        file_size=len(file_bytes),
        mime_type=mime_type,
        status="processing",
    )
    session.add(doc)
    await session.flush()  # Get the ID

    # Save file to storage
    save_uploaded_file(doc.id, filename, file_bytes)

    await session.commit()
    return doc, True
