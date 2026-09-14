"""
DocuLens AI — API routes
"""
import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import get_session, async_session
from backend.database.models import (
    Document, Page, Block, Entity, TableExtraction,
    Insight, TimelineEvent, Correction, ProcessingJob,
)
from backend.ingestion.service import ingest_document, IngestionError
from backend.pipeline.orchestrator import run_pipeline, get_events
from backend.config import CONFIDENCE_THRESHOLD, DOCUMENTS_DIR
from backend.models.schemas import (
    DocumentOut, EntityOut, EntityUpdate, TableOut,
    InsightOut, TimelineEventOut, PageOut, BlockOut,
    BBox, EvidenceRef, ProcessingStageOut, DocumentFullOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ─── Helper: ORM → Pydantic ──────────────────────────────
def _doc_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        document_type=doc.document_type,
        status=doc.status,
        overall_confidence=doc.overall_confidence,
        page_count=doc.page_count,
        is_scanned=doc.is_scanned or False,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
    )


def _entity_out(e: Entity) -> EntityOut:
    bbox = None
    if e.bbox_x0 is not None:
        bbox = BBox(x0=e.bbox_x0, y0=e.bbox_y0, x1=e.bbox_x1, y1=e.bbox_y1)
    return EntityOut(
        id=e.id,
        entity_type=e.entity_type,
        value=e.value,
        page=e.page,
        block_id=e.block_id,
        confidence=e.confidence,
        verified=e.verified or False,
        evidence_text=e.evidence_text,
        bbox=bbox,
    )


def _table_out(t: TableExtraction) -> TableOut:
    bbox = None
    if t.bbox_x0 is not None:
        bbox = BBox(x0=t.bbox_x0, y0=t.bbox_y0, x1=t.bbox_x1, y1=t.bbox_y1)
    return TableOut(
        id=t.id,
        page=t.page,
        title=t.title,
        columns=json.loads(t.columns_json) if t.columns_json else [],
        rows=json.loads(t.rows_json) if t.rows_json else [],
        confidence=t.confidence,
        bbox=bbox,
    )


def _insight_out(ins: Insight) -> InsightOut:
    evidence = []
    if ins.evidence_json:
        try:
            for ref in json.loads(ins.evidence_json):
                evidence.append(EvidenceRef(
                    page=ref.get("page", 0),
                    block_id=ref.get("block_id"),
                    text=ref.get("text"),
                ))
        except Exception:
            pass
    return InsightOut(
        id=ins.id,
        insight_type=ins.insight_type,
        severity=ins.severity,
        title=ins.title,
        description=ins.description,
        evidence=evidence,
        confidence=ins.confidence,
    )


def _timeline_out(te: TimelineEvent) -> TimelineEventOut:
    evidence = []
    if te.evidence_json:
        try:
            for ref in json.loads(te.evidence_json):
                evidence.append(EvidenceRef(
                    page=ref.get("page", 0),
                    text=ref.get("text"),
                ))
        except Exception:
            pass
    return TimelineEventOut(
        id=te.id,
        date=te.date,
        event=te.event,
        evidence=evidence,
        confidence=te.confidence,
    )


# ─── Upload ──────────────────────────────────────────────
@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    file_bytes = await file.read()

    try:
        doc, is_new = await ingest_document(session, file.filename, file_bytes)
    except IngestionError as e:
        raise HTTPException(400, str(e))

    if not is_new:
        return {
            "document": _doc_out(doc).model_dump(),
            "cached": True,
            "message": "Cached result found — retrieved instantly",
        }

    # Start async processing pipeline in background
    asyncio.create_task(_run_pipeline_bg(doc.id))

    return {
        "document": _doc_out(doc).model_dump(),
        "cached": False,
        "message": "Processing started",
    }


async def _run_pipeline_bg(doc_id: str):
    """Run the pipeline in a background task with its own dedicated session."""
    async with async_session() as session:
        try:
            await run_pipeline(doc_id, session)
        except Exception as e:
            logger.error(f"Pipeline background error for {doc_id}: {e}")


# ─── List Documents ──────────────────────────────────────
@router.get("/documents")
async def list_documents(session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return {"documents": [_doc_out(d).model_dump() for d in docs]}


# ─── Get Document ────────────────────────────────────────
@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Fetch all related data
    pages_q = await session.execute(
        select(Page).where(Page.document_id == doc_id).order_by(Page.page_number)
    )
    pages = pages_q.scalars().all()

    entities_q = await session.execute(
        select(Entity).where(Entity.document_id == doc_id)
    )
    entities = entities_q.scalars().all()

    tables_q = await session.execute(
        select(TableExtraction).where(TableExtraction.document_id == doc_id)
    )
    tables = tables_q.scalars().all()

    insights_q = await session.execute(
        select(Insight).where(Insight.document_id == doc_id)
    )
    insights = insights_q.scalars().all()

    timeline_q = await session.execute(
        select(TimelineEvent).where(TimelineEvent.document_id == doc_id).order_by(TimelineEvent.date)
    )
    timeline = timeline_q.scalars().all()

    jobs_q = await session.execute(
        select(ProcessingJob).where(ProcessingJob.document_id == doc_id).order_by(ProcessingJob.created_at)
    )
    jobs = jobs_q.scalars().all()

    # Build page output with blocks
    pages_out = []
    for p in pages:
        blocks_q = await session.execute(
            select(Block).where(Block.page_id == p.id).order_by(Block.order_index)
        )
        blocks = blocks_q.scalars().all()
        pages_out.append(PageOut(
            id=p.id,
            page_number=p.page_number,
            image_url=p.image_path,
            width=p.width,
            height=p.height,
            text=p.text,
            ocr_confidence=p.ocr_confidence,
            blocks=[BlockOut(
                id=b.id,
                block_type=b.block_type,
                text=b.text,
                bbox=BBox(x0=b.bbox_x0 or 0, y0=b.bbox_y0 or 0, x1=b.bbox_x1 or 1, y1=b.bbox_y1 or 1),
                confidence=b.confidence,
                order_index=b.order_index,
            ) for b in blocks],
        ))

    review_items = [_entity_out(e) for e in entities if e.confidence is not None and e.confidence < CONFIDENCE_THRESHOLD]

    return DocumentFullOut(
        document=_doc_out(doc),
        pages=[p.model_dump() for p in pages_out],
        entities=[_entity_out(e).model_dump() for e in entities],
        tables=[_table_out(t).model_dump() for t in tables],
        insights=[_insight_out(i).model_dump() for i in insights],
        timeline=[_timeline_out(te).model_dump() for te in timeline],
        processing_stages=[ProcessingStageOut(
            stage=j.stage, status=j.status, progress=j.progress, message=j.message,
        ).model_dump() for j in jobs],
        review_items=[r.model_dump() for r in review_items],
    ).model_dump()


# ─── SSE Stream ──────────────────────────────────────────
@router.get("/documents/{doc_id}/stream")
async def stream_processing(doc_id: str):
    """Server-Sent Events stream with database fallback to prevent indefinite waiting."""
    async def event_generator():
        last_index = 0
        iteration = 0
        while True:
            events = get_events(doc_id)
            new_events = events[last_index:]
            for event in new_events:
                yield f"data: {json.dumps(event)}\n\n"
                last_index += 1
                if event.get("stage") in ("complete", "error") or event.get("status") in ("completed", "failed"):
                    return

            iteration += 1

            # Check database state periodically or if event list is empty
            if iteration % 4 == 0 or (iteration == 1 and not events):
                async with async_session() as s:
                    doc = await s.get(Document, doc_id)
                    if doc and doc.status in ("completed", "failed"):
                        terminal_stage = "complete" if doc.status == "completed" else "error"
                        yield f"data: {json.dumps({'stage': terminal_stage, 'status': doc.status, 'progress': 100, 'message': f'Document {doc.status}'})}\n\n"
                        return

            # Keep-alive heartbeat comment every 5 seconds
            if iteration % 10 == 0:
                yield ": ping\n\n"

            # Maximum timeout ~ 120 seconds
            if iteration > 240:
                yield f"data: {json.dumps({'stage': 'error', 'status': 'failed', 'progress': 0, 'message': 'Processing stream timed out'})}\n\n"
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Page Images ─────────────────────────────────────────
@router.get("/documents/{doc_id}/pages/{page_num}/image")
async def get_page_image(doc_id: str, page_num: int, session: AsyncSession = Depends(get_session)):
    """Serve a rendered page image, generating a clean text preview if not on disk."""
    pages_dir = DOCUMENTS_DIR / doc_id / "pages"
    for fmt in ("png", "jpg", "jpeg"):
        img_path = pages_dir / f"page_{page_num:03d}.{fmt}"
        if img_path.exists():
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[fmt]
            return FileResponse(str(img_path), media_type=mime)

    # If image does not exist, fetch page text from database and synthesize preview
    p_res = await session.execute(
        select(Page).where(Page.document_id == doc_id, Page.page_number == page_num)
    )
    page_rec = p_res.scalars().first()
    text = page_rec.text if page_rec and page_rec.text else f"Page {page_num}"

    from backend.ocr.service import _generate_text_canvas_image
    fallback_bytes = _generate_text_canvas_image(text, page_num=page_num)
    return Response(content=fallback_bytes, media_type="image/png")


# ─── Serve original document ─────────────────────────────
@router.get("/documents/{doc_id}/original")
async def get_original_document(doc_id: str):
    """Serve the original uploaded document for PDF viewer."""
    doc_dir = DOCUMENTS_DIR / doc_id
    if not doc_dir.exists():
        raise HTTPException(404, "Document not found")
    for f in doc_dir.iterdir():
        if f.stem == "original":
            mime_map = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
            return FileResponse(str(f), media_type=mime_map.get(f.suffix.lower(), "application/octet-stream"))
    raise HTTPException(404, "Original file not found")


# ─── Entity Correction (Human-in-the-loop) ───────────────
@router.patch("/entities/{entity_id}")
async def correct_entity(
    entity_id: str,
    update: EntityUpdate,
    session: AsyncSession = Depends(get_session),
):
    entity = await session.get(Entity, entity_id)
    if not entity:
        raise HTTPException(404, "Entity not found")

    # Save correction record
    correction = Correction(
        entity_id=entity_id,
        old_value=entity.value,
        new_value=update.value,
    )
    session.add(correction)

    # Update entity
    entity.value = update.value
    entity.verified = update.verified
    entity.confidence = 1.0  # Human-verified = max confidence

    await session.commit()

    return {"entity": _entity_out(entity).model_dump(), "corrected": True}


# ─── Reprocess Document ──────────────────────────────────
@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Reprocess a failed or incomplete document from scratch."""
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Clean existing extracted records
    await session.execute(delete(Entity).where(Entity.document_id == doc_id))
    await session.execute(delete(TableExtraction).where(TableExtraction.document_id == doc_id))
    await session.execute(delete(Insight).where(Insight.document_id == doc_id))
    await session.execute(delete(TimelineEvent).where(TimelineEvent.document_id == doc_id))
    
    # Clean blocks and pages
    page_ids_q = await session.execute(select(Page.id).where(Page.document_id == doc_id))
    page_ids = page_ids_q.scalars().all()
    if page_ids:
        await session.execute(delete(Block).where(Block.page_id.in_(page_ids)))
    await session.execute(delete(Page).where(Page.document_id == doc_id))
    await session.execute(delete(ProcessingJob).where(ProcessingJob.document_id == doc_id))

    doc.status = "processing"
    doc.overall_confidence = None
    doc.document_type = None
    await session.commit()

    # Clear page images directory
    pages_dir = DOCUMENTS_DIR / doc_id / "pages"
    if pages_dir.exists():
        import shutil
        shutil.rmtree(pages_dir, ignore_errors=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Re-launch pipeline
    asyncio.create_task(_run_pipeline_bg(doc_id))

    return {
        "document": _doc_out(doc).model_dump(),
        "message": "Reprocessing started",
    }


# ─── Delete Document ─────────────────────────────────────
@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a document and its stored files."""
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from backend.storage.file_store import delete_document_files
    delete_document_files(doc_id)

    await session.delete(doc)
    await session.commit()

    return {"message": "Document deleted successfully", "id": doc_id}


# ─── Review Queue ────────────────────────────────────────
@router.get("/documents/{doc_id}/review")
async def get_review_items(doc_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Entity).where(
            Entity.document_id == doc_id,
            Entity.confidence < CONFIDENCE_THRESHOLD,
        )
    )
    entities = result.scalars().all()
    return {"review_items": [_entity_out(e).model_dump() for e in entities]}


# ─── Entities ────────────────────────────────────────────
@router.get("/documents/{doc_id}/entities")
async def get_entities(doc_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Entity).where(Entity.document_id == doc_id)
    )
    return {"entities": [_entity_out(e).model_dump() for e in result.scalars().all()]}


# ─── Tables ──────────────────────────────────────────────
@router.get("/documents/{doc_id}/tables")
async def get_tables(doc_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(TableExtraction).where(TableExtraction.document_id == doc_id)
    )
    return {"tables": [_table_out(t).model_dump() for t in result.scalars().all()]}


# ─── Insights ────────────────────────────────────────────
@router.get("/documents/{doc_id}/insights")
async def get_insights(doc_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Insight).where(Insight.document_id == doc_id)
    )
    return {"insights": [_insight_out(i).model_dump() for i in result.scalars().all()]}


# ─── Timeline ───────────────────────────────────────────
@router.get("/documents/{doc_id}/timeline")
async def get_timeline(doc_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(TimelineEvent).where(TimelineEvent.document_id == doc_id).order_by(TimelineEvent.date)
    )
    return {"timeline": [_timeline_out(te).model_dump() for te in result.scalars().all()]}
