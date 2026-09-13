"""
DocuLens AI — Pydantic schemas for API and inter-service communication
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Shared ──────────────────────────────────────────────
class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class EvidenceRef(BaseModel):
    page: int
    block_id: str | None = None
    text: str | None = None


# ─── Document ────────────────────────────────────────────
class DocumentOut(BaseModel):
    id: str
    filename: str
    document_type: str | None = None
    status: str
    overall_confidence: float | None = None
    page_count: int | None = None
    is_scanned: bool = False
    created_at: str


class DocumentListOut(BaseModel):
    documents: list[DocumentOut]


# ─── Page ─────────────────────────────────────────────────
class BlockOut(BaseModel):
    id: str
    block_type: str
    text: str | None = None
    bbox: BBox | None = None
    confidence: float | None = None
    order_index: int | None = None


class PageOut(BaseModel):
    id: str
    page_number: int
    image_url: str | None = None
    width: int | None = None
    height: int | None = None
    text: str | None = None
    ocr_confidence: float | None = None
    blocks: list[BlockOut] = []


# ─── Entity ──────────────────────────────────────────────
class EntityOut(BaseModel):
    id: str
    entity_type: str
    value: str
    page: int | None = None
    block_id: str | None = None
    confidence: float | None = None
    verified: bool = False
    evidence_text: str | None = None
    bbox: BBox | None = None


class EntityUpdate(BaseModel):
    value: str
    verified: bool = True


# ─── Table ───────────────────────────────────────────────
class TableOut(BaseModel):
    id: str
    page: int
    title: str | None = None
    columns: list[str] = []
    rows: list[list] = []
    confidence: float | None = None
    bbox: BBox | None = None


# ─── Insight ─────────────────────────────────────────────
class InsightOut(BaseModel):
    id: str
    insight_type: str
    severity: str | None = None
    title: str
    description: str
    evidence: list[EvidenceRef] = []
    confidence: float | None = None


# ─── Timeline ───────────────────────────────────────────
class TimelineEventOut(BaseModel):
    id: str
    date: str
    event: str
    evidence: list[EvidenceRef] = []
    confidence: float | None = None


# ─── Processing ─────────────────────────────────────────
class ProcessingStageOut(BaseModel):
    stage: str
    status: str
    progress: int = 0
    message: str | None = None


class ProcessingStreamEvent(BaseModel):
    """Sent via SSE."""
    stage: str
    status: str
    progress: int = 0
    message: str | None = None
    document_id: str | None = None


# ─── Full document result ────────────────────────────────
class DocumentFullOut(BaseModel):
    document: DocumentOut
    pages: list[PageOut] = []
    entities: list[EntityOut] = []
    tables: list[TableOut] = []
    insights: list[InsightOut] = []
    timeline: list[TimelineEventOut] = []
    processing_stages: list[ProcessingStageOut] = []
    review_items: list[EntityOut] = []  # entities below confidence threshold


# ─── Gemini structured output schemas ────────────────────
class GeminiClassification(BaseModel):
    document_type: str = Field(description="One of: invoice, research_paper, report, medical_report, other")
    confidence: float = Field(description="Confidence between 0 and 1")
    signals: list[str] = Field(description="Key phrases that led to this classification")


class GeminiBlock(BaseModel):
    block_type: str = Field(description="One of: heading, text, table, image, key_value")
    text: str
    bbox: list[float] = Field(description="[x0, y0, x1, y1] normalized 0-1")
    confidence: float


class GeminiBlockExtraction(BaseModel):
    blocks: list[GeminiBlock] = []


class GeminiEntityExtraction(BaseModel):
    entities: list[GeminiEntity] = []


class GeminiEntity(BaseModel):
    entity_type: str = Field(description="e.g. PERSON, DATE, AMOUNT, VENDOR, INVOICE_NUMBER, etc.")
    value: str
    page: int
    source_text: str = Field(description="The exact text from the document that contains this entity")
    confidence: float


class GeminiTableExtraction(BaseModel):
    tables: list[GeminiTable] = []


class GeminiTable(BaseModel):
    title: str | None = None
    page: int
    columns: list[str]
    rows: list[list[str]]
    confidence: float


class GeminiInsights(BaseModel):
    timeline_events: list[GeminiTimelineEvent] = []
    anomalies: list[GeminiInsight] = []
    findings: list[GeminiInsight] = []
    missing_information: list[GeminiInsight] = []
    contradictions: list[GeminiInsight] = []


class GeminiTimelineEvent(BaseModel):
    date: str
    event: str
    page: int
    confidence: float


class GeminiInsight(BaseModel):
    insight_type: str
    severity: str = Field(description="One of: high, medium, low")
    title: str
    description: str
    evidence_pages: list[int] = []
    evidence_text: list[str] = []
    confidence: float
