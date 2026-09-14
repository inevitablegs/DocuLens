"""
DocuLens AI — Pipeline orchestrator
Runs the full document processing pipeline asynchronously and emits SSE events.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import (
    Document, Page, Block, Entity, TableExtraction,
    Insight, TimelineEvent, ProcessingJob,
)
from backend.storage.file_store import get_original_path
from backend.classifier.service import classify
from backend.ocr.service import (
    analyze_document_quality,
    extract_from_digital_pdf,
    extract_from_image,
    extract_from_scanned_pdf,
)
from backend.gemini_client import extract_entities, extract_tables, generate_insights
from backend.confidence.service import (
    compute_entity_confidence,
    compute_document_confidence,
    compute_evidence_match_score,
    needs_human_review,
)

logger = logging.getLogger(__name__)

# In-memory event store for SSE streaming (per document)
_events: dict[str, list[dict]] = {}


def get_events(doc_id: str) -> list[dict]:
    return _events.get(doc_id, [])


def _emit(doc_id: str, stage: str, status: str, progress: int, message: str = ""):
    event = {
        "stage": stage,
        "status": status,
        "progress": progress,
        "message": message,
        "document_id": doc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if doc_id not in _events:
        _events[doc_id] = []
    _events[doc_id].append(event)


async def _save_processing_job(session: AsyncSession, doc_id: str, stage: str, status: str, progress: int, message: str):
    job = ProcessingJob(
        document_id=doc_id,
        stage=stage,
        status=status,
        progress=progress,
        message=message,
    )
    if status == "completed":
        job.completed_at = datetime.now(timezone.utc)
    session.add(job)
    await session.commit()


async def run_pipeline(doc_id: str, session: AsyncSession):
    """Run the full document processing pipeline with stage commits and resilient error handling."""
    _events[doc_id] = []  # Reset events

    try:
        doc = await session.get(Document, doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        doc.status = "processing"
        await session.commit()

        # ──── Stage 1: Ingestion ────────────────────────────
        _emit(doc_id, "ingestion", "running", 50, "Validating document...")
        file_path = get_original_path(doc_id)
        if not file_path or not file_path.exists():
            raise FileNotFoundError("Original uploaded file not found on disk")

        _emit(doc_id, "ingestion", "completed", 100, "Document validated")
        await _save_processing_job(session, doc_id, "ingestion", "completed", 100, "Document validated")

        # ──── Stage 2: Quality Analysis + Routing ───────────
        _emit(doc_id, "routing", "running", 50, "Analyzing document quality...")
        quality = await analyze_document_quality(file_path)
        page_count = max(1, quality.get("page_count", 1))
        is_scanned = quality.get("is_scanned", False)
        pipeline_type = quality.get("recommended_pipeline", ["pdf_parser"])[0]

        # Update document
        doc.page_count = page_count
        doc.is_scanned = is_scanned
        await session.commit()

        _emit(doc_id, "routing", "completed", 100,
              f"{'Scanned' if is_scanned else 'Digital'} document · {page_count} pages · Pipeline: {pipeline_type}")
        await _save_processing_job(session, doc_id, "routing", "completed", 100,
                                   f"Pipeline: {pipeline_type}")

        # ──── Stage 3: OCR / Text Extraction ────────────────
        _emit(doc_id, "ocr", "running", 0, "Extracting text...")

        if file_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            ocr_result = await extract_from_image(file_path, doc_id)
        elif is_scanned:
            ocr_result = await extract_from_scanned_pdf(file_path, doc_id)
        else:
            ocr_result = await extract_from_digital_pdf(file_path, doc_id)

        # Save pages and blocks to DB
        page_texts: dict[int, str] = {}
        page_ocr_confidences: list[float] = []
        blocks_by_page: dict[int, list[dict]] = {}

        pages_list = ocr_result.get("pages", [])
        if not pages_list:
            # Generate at least one fallback page
            pages_list = [{
                "page_number": 1,
                "width": 800,
                "height": 1050,
                "text": "No text could be extracted.",
                "ocr_confidence": 0.5,
                "blocks": [],
            }]

        for p_data in pages_list:
            page_num = p_data["page_number"]
            page_text = p_data.get("text", "")
            page_texts[page_num] = page_text
            ocr_conf = p_data.get("ocr_confidence", 0.9)
            page_ocr_confidences.append(ocr_conf)
            blocks_by_page[page_num] = []

            page = Page(
                document_id=doc_id,
                page_number=page_num,
                image_path=f"/api/documents/{doc_id}/pages/{page_num}/image",
                width=p_data.get("width", 800),
                height=p_data.get("height", 1050),
                text=page_text,
                ocr_confidence=ocr_conf,
            )
            session.add(page)
            await session.flush()

            # Save blocks (DIR)
            for b_data in p_data.get("blocks", []):
                bbox = b_data.get("bbox", [0, 0, 1, 1])
                block = Block(
                    page_id=page.id,
                    block_type=b_data.get("block_type", "text"),
                    text=b_data.get("text", ""),
                    bbox_x0=bbox[0] if len(bbox) > 0 else 0,
                    bbox_y0=bbox[1] if len(bbox) > 1 else 0,
                    bbox_x1=bbox[2] if len(bbox) > 2 else 1,
                    bbox_y1=bbox[3] if len(bbox) > 3 else 1,
                    confidence=b_data.get("confidence", 0.9),
                    order_index=b_data.get("order_index", 0),
                )
                session.add(block)
                blocks_by_page[page_num].append({
                    "id": block.id,
                    "text": block.text,
                    "bbox": [block.bbox_x0, block.bbox_y0, block.bbox_x1, block.bbox_y1],
                })

            progress = int((page_num / max(len(pages_list), 1)) * 100)
            _emit(doc_id, "ocr", "running", progress, f"Processed page {page_num}/{len(pages_list)}")

        doc.page_count = len(pages_list)
        await session.commit()

        _emit(doc_id, "ocr", "completed", 100,
              f"Text extracted from {len(pages_list)} page(s) via {ocr_result.get('method', 'unknown')}")
        await _save_processing_job(session, doc_id, "ocr", "completed", 100, "Text extraction complete")

        # ──── Stage 4: Classification ───────────────────────
        _emit(doc_id, "classification", "running", 50, "Classifying document...")

        all_text = " ".join(page_texts.values()).strip()
        classification = await classify(all_text[:8000])

        doc.document_type = classification.get("document_type", "report")
        doc.classification_signals = json.dumps(classification.get("signals", []))
        await session.commit()

        _emit(doc_id, "classification", "completed", 100,
              f"Type: {doc.document_type} · Confidence: {classification.get('confidence', 0):.0%} · Method: {classification.get('method', 'unknown')}")
        await _save_processing_job(session, doc_id, "classification", "completed", 100,
                                   f"{doc.document_type} ({classification.get('method')})")

        # ──── Stage 5: Entity Extraction ────────────────────
        _emit(doc_id, "extraction", "running", 30, "Extracting entities...")

        try:
            entity_result = await extract_entities(all_text, doc.document_type, page_texts)
            raw_entities = entity_result.get("entities", [])
        except Exception as e:
            logger.warning(f"Entity extraction error: {e}")
            raw_entities = []

        _emit(doc_id, "extraction", "running", 60, f"Found {len(raw_entities)} entities. Extracting tables...")

        # ──── Stage 5b: Table Extraction ────────────────────
        try:
            table_result = await extract_tables(all_text, doc.document_type, page_texts)
            raw_tables = table_result.get("tables", [])
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")
            raw_tables = []

        # Save entities with confidence scoring and block/bbox resolution
        avg_ocr = sum(page_ocr_confidences) / len(page_ocr_confidences) if page_ocr_confidences else 0.9
        entity_confidences = []

        for e_data in raw_entities:
            evidence_score = compute_evidence_match_score(
                e_data.get("value", ""),
                e_data.get("source_text", ""),
            )

            confidence = compute_entity_confidence(
                ocr_confidence=avg_ocr,
                extraction_confidence=e_data.get("confidence", 0.9),
                schema_valid=bool(e_data.get("value")),
                evidence_match_score=evidence_score,
                model_confidence=e_data.get("confidence", 0.9),
            )

            # Match entity to block and bounding box
            target_val = (e_data.get("value") or "").strip().lower()
            target_src = (e_data.get("source_text") or "").strip().lower()
            e_page = e_data.get("page") or 1
            matched_bbox = e_data.get("bbox")
            matched_block_id = None

            candidate_pages = [e_page] if e_page in blocks_by_page else list(blocks_by_page.keys())
            for p_num in candidate_pages:
                for b_info in blocks_by_page.get(p_num, []):
                    b_txt = (b_info["text"] or "").lower()
                    if (target_val and target_val in b_txt) or (target_src and (target_src in b_txt or b_txt in target_src)):
                        matched_bbox = b_info["bbox"]
                        matched_block_id = b_info["id"]
                        e_page = p_num
                        break
                if matched_block_id:
                    break

            bbox_x0 = matched_bbox[0] if matched_bbox and len(matched_bbox) > 0 else None
            bbox_y0 = matched_bbox[1] if matched_bbox and len(matched_bbox) > 1 else None
            bbox_x1 = matched_bbox[2] if matched_bbox and len(matched_bbox) > 2 else None
            bbox_y1 = matched_bbox[3] if matched_bbox and len(matched_bbox) > 3 else None

            entity = Entity(
                document_id=doc_id,
                entity_type=e_data.get("entity_type", "UNKNOWN"),
                value=e_data.get("value", ""),
                page=e_page,
                block_id=matched_block_id,
                bbox_x0=bbox_x0,
                bbox_y0=bbox_y0,
                bbox_x1=bbox_x1,
                bbox_y1=bbox_y1,
                confidence=confidence,
                verified=not needs_human_review(confidence),
                evidence_text=e_data.get("source_text", ""),
            )
            session.add(entity)
            entity_confidences.append(confidence)

        # Save tables
        for t_data in raw_tables:
            table = TableExtraction(
                document_id=doc_id,
                page=t_data.get("page", 1),
                title=t_data.get("title"),
                columns_json=json.dumps(t_data.get("columns", [])),
                rows_json=json.dumps(t_data.get("rows", [])),
                confidence=t_data.get("confidence", 0.9),
            )
            session.add(table)

        await session.commit()

        _emit(doc_id, "extraction", "completed", 100,
              f"Extracted {len(raw_entities)} entities and {len(raw_tables)} tables")
        await _save_processing_job(session, doc_id, "extraction", "completed", 100,
                                   f"{len(raw_entities)} entities, {len(raw_tables)} tables")

        # ──── Stage 6: Insight Generation ───────────────────
        _emit(doc_id, "insights", "running", 50, "Generating insights...")

        try:
            entity_dicts = [{"type": e.get("entity_type"), "value": e.get("value")} for e in raw_entities]
            insights_result = await generate_insights(all_text, doc.document_type, entity_dicts, page_texts)
        except Exception as e:
            logger.warning(f"Insights generation error: {e}")
            insights_result = {}

        # Save timeline events
        for te in insights_result.get("timeline_events", []):
            event = TimelineEvent(
                document_id=doc_id,
                date=te.get("date", "unknown"),
                event=te.get("event", ""),
                evidence_json=json.dumps([{"page": te.get("page", 1)}]),
                confidence=te.get("confidence", 0.8),
            )
            session.add(event)

        # Save insights (anomalies, findings, missing, contradictions)
        for category in ["anomalies", "findings", "missing_information", "contradictions"]:
            for ins in insights_result.get(category, []):
                insight = Insight(
                    document_id=doc_id,
                    insight_type=ins.get("insight_type", category.rstrip("s")),
                    severity=ins.get("severity", "medium"),
                    title=ins.get("title", ""),
                    description=ins.get("description", ""),
                    evidence_json=json.dumps([
                        {"page": p, "text": t}
                        for p, t in zip(
                            ins.get("evidence_pages", []),
                            ins.get("evidence_text", [])
                        )
                    ]),
                    confidence=ins.get("confidence", 0.8),
                )
                session.add(insight)

        await session.commit()

        _emit(doc_id, "insights", "completed", 100, "Insights generated")
        await _save_processing_job(session, doc_id, "insights", "completed", 100, "Insights generated")

        # ──── Stage 7: Verification + Final Confidence ──────
        _emit(doc_id, "verification", "running", 50, "Verifying evidence...")

        if entity_confidences:
            doc.overall_confidence = compute_document_confidence(entity_confidences)
        elif page_ocr_confidences:
            doc.overall_confidence = round(sum(page_ocr_confidences) / len(page_ocr_confidences) * 0.9, 4)
        else:
            doc.overall_confidence = 0.85

        review_count = sum(1 for c in entity_confidences if needs_human_review(c))

        _emit(doc_id, "verification", "completed", 100,
              f"Overall confidence: {doc.overall_confidence:.0%} · {review_count} items need review")
        await _save_processing_job(session, doc_id, "verification", "completed", 100,
                                   f"Confidence: {doc.overall_confidence:.0%}")

        # ──── Complete ──────────────────────────────────────
        doc.status = "completed"
        await session.commit()

        _emit(doc_id, "complete", "completed", 100, "Processing complete!")

    except Exception as e:
        logger.exception(f"Pipeline error for {doc_id}: {e}")
        _emit(doc_id, "error", "failed", 0, f"Processing failed: {str(e)}")
        try:
            await session.rollback()
            failed_doc = await session.get(Document, doc_id)
            if failed_doc:
                failed_doc.status = "failed"
                await _save_processing_job(session, doc_id, "pipeline", "failed", 0, str(e))
                await session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to record failure status for {doc_id}: {inner_e}")
        raise
