"""
DocuLens AI — Comprehensive Test Suite
Tests:
- File Ingestion & Deduplication
- Rule-based Classification & Signals
- Extraction with Page & Provenance
- Math Anomaly Detection & Insights
- Composite Confidence Scoring
- Human-in-the-loop (HITL) Correction Loop
"""
import pytest
import asyncio
from pathlib import Path
from starlette.testclient import TestClient

from backend.confidence.service import (
    compute_entity_confidence,
    compute_document_confidence,
    compute_evidence_match_score,
    needs_human_review,
)
from backend.classifier.service import rule_based_classify
from backend.extraction.service import extract_entities_rule_based, extract_tables_rule_based
from backend.insights.service import generate_insights_rule_based
from backend.main import app
from backend.database.db import init_db


@pytest.fixture(scope="module", autouse=True)
def init_database():
    """Ensure DB tables exist before tests run."""
    asyncio.run(init_db())


# ─── 1. Confidence Service Tests ─────────────────────────
def test_confidence_scoring():
    conf = compute_entity_confidence(
        ocr_confidence=1.0,
        extraction_confidence=0.9,
        schema_valid=True,
        evidence_match_score=1.0,
        model_confidence=0.95,
    )
    assert 0.90 <= conf <= 1.0
    assert not needs_human_review(conf)

    low_conf = compute_entity_confidence(
        ocr_confidence=0.5,
        extraction_confidence=0.4,
        schema_valid=False,
        evidence_match_score=0.2,
        model_confidence=0.5,
    )
    assert low_conf < 0.85
    assert needs_human_review(low_conf)


def test_evidence_match_score():
    score_exact = compute_evidence_match_score("Acme Corp", "Invoice billed by Acme Corp for services")
    assert score_exact == 1.0

    score_empty = compute_evidence_match_score("", "some text")
    assert score_empty == 0.0

    score_partial = compute_evidence_match_score("Acme Industrial Corp", "Acme Industrial")
    assert score_partial > 0.4


# ─── 2. Classification Tests ──────────────────────────────
def test_rule_based_classification_invoice():
    sample = """
    INVOICE
    Invoice Number: INV-2026-0042
    Invoice Date: 2026-04-15
    Due Date: 2026-05-15
    Bill To: Global Logistics LLC
    Subtotal: $1,250.00
    Tax: $125.00
    Total Amount: $1,375.00
    """
    doc_type, confidence, signals = rule_based_classify(sample)
    assert doc_type == "invoice"
    assert confidence >= 0.70
    assert "invoice number" in signals or "invoice date" in signals


def test_rule_based_classification_medical():
    sample = """
    Patient Name: Jane Doe
    Patient ID: MRN-99482
    Attending Physician: Dr. Marcus Vance
    Primary Diagnosis: Acute Bronchitis
    Prescription: Amoxicillin 500mg daily
    """
    doc_type, confidence, signals = rule_based_classify(sample)
    assert doc_type == "medical_report"
    assert confidence >= 0.70


# ─── 3. Entity & Table Extraction Tests ───────────────────
def test_invoice_entity_extraction():
    sample = """
    ACME SERVICES INC.
    Invoice #: INV-9901
    Invoice Date: 2026-05-01
    Due Date: 2026-06-01
    Subtotal: $4,500.00
    Tax: $450.00
    Total Amount: $4,950.00
    """
    res = extract_entities_rule_based(sample, "invoice", {1: sample})
    entities = {e["entity_type"]: e["value"] for e in res["entities"]}

    assert "INVOICE_NUMBER" in entities
    assert "INV-9901" in entities["INVOICE_NUMBER"]
    assert "TOTAL_AMOUNT" in entities
    assert "4,950.00" in entities["TOTAL_AMOUNT"]
    assert "SUBTOTAL" in entities
    assert "4,500.00" in entities["SUBTOTAL"]


def test_table_extraction():
    sample = """
    | Item | Quantity | Unit Price | Total |
    | Server Rack | 2 | $1,200.00 | $2,400.00 |
    | Patch Cable | 10 | $15.00 | $150.00 |
    """
    res = extract_tables_rule_based(sample, "invoice", {1: sample})
    assert len(res["tables"]) >= 1
    table = res["tables"][0]
    assert len(table["columns"]) >= 3
    assert len(table["rows"]) >= 2


# ─── 4. Insight Engine & Anomaly Detection ────────────────
def test_anomaly_detection_math_mismatch():
    sample = """
    INVOICE
    Subtotal: $1,000.00
    Tax: $100.00
    Total Amount: $1,500.00
    """
    entities = [
        {"entity_type": "SUBTOTAL", "value": "1,000.00", "page": 1, "source_text": "Subtotal: $1,000.00"},
        {"entity_type": "TAX", "value": "100.00", "page": 1, "source_text": "Tax: $100.00"},
        {"entity_type": "TOTAL_AMOUNT", "value": "1,500.00", "page": 1, "source_text": "Total Amount: $1,500.00"},
    ]
    insights = generate_insights_rule_based(sample, "invoice", entities, {1: sample})
    anomalies = insights["anomalies"]

    assert len(anomalies) >= 1
    assert any("Mismatch" in a["title"] for a in anomalies)
    assert anomalies[0]["severity"] == "high"


def test_missing_fields_detection():
    # Invoice without Due Date or Tax
    sample = "Invoice Number: INV-001\nTotal Amount: $500.00"
    entities = [
        {"entity_type": "INVOICE_NUMBER", "value": "INV-001"},
        {"entity_type": "TOTAL_AMOUNT", "value": "500.00"},
    ]
    insights = generate_insights_rule_based(sample, "invoice", entities, {1: sample})
    missing = [m["title"] for m in insights["missing_information"]]

    assert any("DUE_DATE" in m for m in missing)
    assert any("VENDOR" in m for m in missing)


# ─── 5. API Client & Endpoints ───────────────────────────
def test_api_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "DocuLens AI"
    assert data["status"] == "running"


def test_api_upload_and_list():
    client = TestClient(app)

    # Test file upload (mock document)
    sample_content = b"""%PDF-1.4
    INVOICE INV-8821
    Date: 2026-03-20
    Total: $250.00
    """
    response = client.post(
        "/api/documents",
        files={"file": ("test_invoice.pdf", sample_content, "application/pdf")},
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "document" in res_data
    doc_id = res_data["document"]["id"]
    assert doc_id is not None

    # Test duplicate detection (cache hit)
    dup_res = client.post(
        "/api/documents",
        files={"file": ("test_invoice.pdf", sample_content, "application/pdf")},
    )
    assert dup_res.status_code == 200

    # Test document list
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["id"] == doc_id for d in docs)


def test_hitl_correction_endpoint():
    """Test Human-in-the-loop correction and confidence bump."""
    client = TestClient(app)
    from backend.database.db import async_session
    from backend.database.models import Document, Entity

    async def _setup_entity():
        async with async_session() as session:
            doc = Document(
                id="doc-test-hitl",
                filename="hitl_doc.pdf",
                file_hash="hash-hitl-123",
                file_size=1024,
                mime_type="application/pdf",
                status="completed",
            )
            session.add(doc)
            entity = Entity(
                id="ent-test-1",
                document_id="doc-test-hitl",
                entity_type="TOTAL_AMOUNT",
                value="$450.00",
                confidence=0.62,
                verified=False,
            )
            session.add(entity)
            await session.commit()

    asyncio.run(_setup_entity())

    # Send human correction
    patch_res = client.patch(
        "/api/entities/ent-test-1",
        json={"value": "$500.00", "verified": True},
    )
    assert patch_res.status_code == 200
    data = patch_res.json()
    assert data["corrected"] is True
    assert data["entity"]["value"] == "$500.00"
    assert data["entity"]["confidence"] == 1.0  # Human-verified = 100% confidence
    assert data["entity"]["verified"] is True
