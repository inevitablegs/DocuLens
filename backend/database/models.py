"""
DocuLens AI — ORM models
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship

from backend.database.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Document ────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    document_type = Column(String, nullable=True)  # invoice, research_paper, report …
    status = Column(String, nullable=False, default="pending")  # pending / processing / completed / failed
    overall_confidence = Column(Float, nullable=True)
    page_count = Column(Integer, nullable=True)
    is_scanned = Column(Boolean, default=False)
    classification_signals = Column(Text, nullable=True)  # JSON list
    created_at = Column(DateTime, default=_now)

    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("TableExtraction", back_populates="document", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="document", cascade="all, delete-orphan")
    timeline_events = relationship("TimelineEvent", back_populates="document", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="document", cascade="all, delete-orphan")


# ─── Page ─────────────────────────────────────────────────
class Page(Base):
    __tablename__ = "pages"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)

    document = relationship("Document", back_populates="pages")
    blocks = relationship("Block", back_populates="page", cascade="all, delete-orphan")


# ─── Block (Document Intermediate Representation) ────────
class Block(Base):
    __tablename__ = "blocks"

    id = Column(String, primary_key=True, default=_uuid)
    page_id = Column(String, ForeignKey("pages.id"), nullable=False)
    block_type = Column(String, nullable=False)  # heading, text, table, image, key_value
    text = Column(Text, nullable=True)
    bbox_x0 = Column(Float, nullable=True)
    bbox_y0 = Column(Float, nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    order_index = Column(Integer, nullable=True)

    page = relationship("Page", back_populates="blocks")


# ─── Entity ──────────────────────────────────────────────
class Entity(Base):
    __tablename__ = "entities"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    entity_type = Column(String, nullable=False)  # PERSON, DATE, AMOUNT, VENDOR, etc.
    value = Column(Text, nullable=False)
    page = Column(Integer, nullable=True)
    block_id = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    verified = Column(Boolean, default=False)
    evidence_text = Column(Text, nullable=True)  # The source text that supports this entity
    bbox_x0 = Column(Float, nullable=True)
    bbox_y0 = Column(Float, nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)

    document = relationship("Document", back_populates="entities")
    corrections = relationship("Correction", back_populates="entity", cascade="all, delete-orphan")


# ─── Table ───────────────────────────────────────────────
class TableExtraction(Base):
    __tablename__ = "tables"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page = Column(Integer, nullable=False)
    title = Column(String, nullable=True)
    columns_json = Column(Text, nullable=True)  # JSON array of column names
    rows_json = Column(Text, nullable=True)  # JSON array of row arrays
    confidence = Column(Float, nullable=True)
    bbox_x0 = Column(Float, nullable=True)
    bbox_y0 = Column(Float, nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)

    document = relationship("Document", back_populates="tables")


# ─── Insight ─────────────────────────────────────────────
class Insight(Base):
    __tablename__ = "insights"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    insight_type = Column(String, nullable=False)  # anomaly, finding, missing, contradiction
    severity = Column(String, nullable=True)  # high, medium, low
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)  # JSON: [{page, block_id, text}]
    confidence = Column(Float, nullable=True)

    document = relationship("Document", back_populates="insights")


# ─── TimelineEvent ───────────────────────────────────────
class TimelineEvent(Base):
    __tablename__ = "timeline_events"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    date = Column(String, nullable=False)  # ISO date string
    event = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)

    document = relationship("Document", back_populates="timeline_events")


# ─── Correction ──────────────────────────────────────────
class Correction(Base):
    __tablename__ = "corrections"

    id = Column(String, primary_key=True, default=_uuid)
    entity_id = Column(String, ForeignKey("entities.id"), nullable=False)
    old_value = Column(Text, nullable=False)
    new_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_now)

    entity = relationship("Entity", back_populates="corrections")


# ─── ProcessingJob ───────────────────────────────────────
class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    stage = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending / running / completed / failed
    progress = Column(Integer, default=0)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now)
    completed_at = Column(DateTime, nullable=True)

    document = relationship("Document", back_populates="processing_jobs")
