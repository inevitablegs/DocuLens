"""
DocuLens AI — Gemini client wrapper with automated fallback and robust JSON parsing
"""
import json
import logging
import re
from typing import Any
from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.extraction.service import extract_entities_rule_based, extract_tables_rule_based
from backend.insights.service import generate_insights_rule_based

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client | None:
    global _client
    if not GEMINI_API_KEY:
        return None
    if _client is None:
        try:
            _client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialize Gemini client: {e}")
            return None
    return _client


def safe_parse_json(text: str | None, default: Any = None) -> Any:
    """
    Robustly extract and parse JSON from model responses.
    Handles markdown fences, leading/trailing commentary, and slight syntax deviations.
    """
    if not text or not isinstance(text, str):
        return default if default is not None else {}

    cleaned = text.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Extract from markdown code fence ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            cleaned = inner

    # 3. Find outermost JSON object { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # Try cleaning trailing commas: e.g. [1, 2,] or {"a": 1,}
            candidate_no_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate_no_commas)
            except Exception:
                pass

    # 4. Find outermost JSON array [ ... ]
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = cleaned[first_bracket:last_bracket + 1]
        try:
            return json.loads(candidate)
        except Exception:
            candidate_no_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate_no_commas)
            except Exception:
                pass

    logger.warning(f"safe_parse_json could not parse model response (snippet: {cleaned[:150]}...)")
    return default if default is not None else {}


async def classify_document(text_sample: str) -> dict:
    """Classify a document using Gemini structured output with fallback."""
    client = get_client()
    if not client:
        return {
            "document_type": "report",
            "confidence": 0.5,
            "signals": ["offline_mode"],
        }

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""Classify this document based on the following text sample.
Determine the document type and provide your confidence level.

Document types: invoice, research_paper, report, medical_report, other

Text sample:
---
{text_sample[:4000]}
---

Respond with JSON: {{"document_type": "...", "confidence": 0.0-1.0, "signals": ["key phrases"]}}""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        parsed = safe_parse_json(response.text)
        if parsed and "document_type" in parsed:
            return parsed
        raise ValueError("Invalid structure returned by Gemini")
    except Exception as e:
        logger.warning(f"Gemini classification failed: {e}. Using fallback.")
        return {
            "document_type": "report",
            "confidence": 0.5,
            "signals": ["fallback"],
        }


async def extract_text_from_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """OCR via Gemini Vision — extract text with layout information."""
    client = get_client()
    if not client:
        return {
            "blocks": [
                {
                    "block_type": "text",
                    "text": "Scanned document (Gemini API key not configured for vision OCR)",
                    "bbox": [0.1, 0.1, 0.9, 0.2],
                    "confidence": 0.5,
                }
            ],
            "full_text": "Scanned document (Gemini API key not configured for vision OCR)",
            "ocr_confidence": 0.5,
        }

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                """Extract ALL text from this document image. Preserve the layout structure.
For each text block, identify its type (heading, text, table, key_value, image_caption).

Respond with JSON:
{
  "blocks": [
    {
      "block_type": "heading|text|table|key_value|image_caption",
      "text": "the extracted text",
      "bbox": [x0, y0, x1, y1],
      "confidence": 0.0-1.0
    }
  ],
  "full_text": "all text concatenated",
  "ocr_confidence": 0.0-1.0
}

Coordinates should be normalized to 0-1 range (fraction of image width/height).
bbox format: [left, top, right, bottom].""",
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        parsed = safe_parse_json(response.text)
        if parsed and ("blocks" in parsed or "full_text" in parsed):
            return parsed
        raise ValueError("Invalid vision OCR response structure")
    except Exception as e:
        logger.warning(f"Gemini vision OCR failed: {e}")
        return {
            "blocks": [],
            "full_text": "",
            "ocr_confidence": 0.0,
        }


async def extract_entities(text: str, document_type: str, page_texts: dict[int, str]) -> dict:
    """Extract structured entities based on document type."""
    client = get_client()
    if not client:
        return extract_entities_rule_based(text, document_type, page_texts)

    type_schemas = {
        "invoice": """Extract: VENDOR, INVOICE_NUMBER, INVOICE_DATE, DUE_DATE, LINE_ITEM, SUBTOTAL, TAX, TOTAL_AMOUNT, CURRENCY, PAYMENT_TERMS, BILLING_ADDRESS, SHIPPING_ADDRESS""",
        "research_paper": """Extract: TITLE, AUTHOR, INSTITUTION, DATE, ABSTRACT_SUMMARY, METHODOLOGY, DATASET, KEY_RESULT, LIMITATION, REFERENCE_COUNT, DOI, JOURNAL""",
        "report": """Extract: TITLE, AUTHOR, ORGANIZATION, DATE, SECTION_TITLE, KEY_FINDING, RECOMMENDATION, METRIC, LOCATION, PERSON, REFERENCE""",
        "medical_report": """Extract: PATIENT_NAME, PATIENT_ID, DATE, DOCTOR, DEPARTMENT, DIAGNOSIS, MEDICATION, DOSAGE, LAB_TEST, LAB_RESULT, VITAL_SIGN, PROCEDURE, FOLLOW_UP_DATE""",
    }

    schema_hint = type_schemas.get(document_type, """Extract: PERSON, ORGANIZATION, DATE, LOCATION, AMOUNT, PRODUCT, REFERENCE, KEY_VALUE""")

    page_context = ""
    for page_num, page_text in sorted(page_texts.items()):
        page_context += f"\n--- PAGE {page_num} ---\n{page_text[:3000]}\n"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""You are a document entity extraction system. Extract structured entities from this {document_type} document.

{schema_hint}

For each entity, provide the exact source text from the document that contains it.

Document text:
{page_context}

Respond with JSON:
{{
  "entities": [
    {{
      "entity_type": "TYPE",
      "value": "extracted value",
      "page": 1,
      "source_text": "exact text from document containing this entity",
      "confidence": 0.0-1.0
    }}
  ]
}}""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        parsed = safe_parse_json(response.text)
        if parsed and "entities" in parsed:
            return parsed
        raise ValueError("Invalid entities response structure")
    except Exception as e:
        logger.warning(f"Gemini entity extraction failed: {e}. Using rule-based fallback.")
        return extract_entities_rule_based(text, document_type, page_texts)


async def extract_tables(text: str, document_type: str, page_texts: dict[int, str]) -> dict:
    """Extract tables from document text."""
    client = get_client()
    if not client:
        return extract_tables_rule_based(text, document_type, page_texts)

    page_context = ""
    for page_num, page_text in sorted(page_texts.items()):
        page_context += f"\n--- PAGE {page_num} ---\n{page_text[:3000]}\n"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""Extract ALL tables from this document. If there are no tables, return an empty list.

Document text:
{page_context}

Respond with JSON:
{{
  "tables": [
    {{
      "title": "table title or null",
      "page": 1,
      "columns": ["col1", "col2"],
      "rows": [["val1", "val2"]],
      "confidence": 0.0-1.0
    }}
  ]
}}""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        parsed = safe_parse_json(response.text)
        if parsed and "tables" in parsed:
            return parsed
        raise ValueError("Invalid tables response structure")
    except Exception as e:
        logger.warning(f"Gemini table extraction failed: {e}. Using rule-based fallback.")
        return extract_tables_rule_based(text, document_type, page_texts)


async def generate_insights(text: str, document_type: str, entities: list[dict], page_texts: dict[int, str]) -> dict:
    """Generate structured insights: timeline, anomalies, findings, missing info, contradictions."""
    client = get_client()
    if not client:
        return generate_insights_rule_based(text, document_type, entities, page_texts)

    entities_summary = json.dumps(entities[:50], indent=2) if entities else "[]"

    page_context = ""
    for page_num, page_text in sorted(page_texts.items()):
        page_context += f"\n--- PAGE {page_num} ---\n{page_text[:2000]}\n"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""You are a document intelligence system. Analyze this {document_type} document and generate structured insights.

Document text:
{page_context}

Extracted entities:
{entities_summary}

Generate:
1. Timeline events (dates and what happened)
2. Anomalies (unusual values, inconsistencies, calculation errors)
3. Key findings (important information)
4. Missing information (expected but not found for a {document_type})
5. Contradictions (conflicting information)

For each insight, reference the specific page and evidence text.

Respond with JSON:
{{
  "timeline_events": [
    {{"date": "YYYY-MM-DD or approximate", "event": "description", "page": 1, "confidence": 0.0-1.0}}
  ],
  "anomalies": [
    {{"insight_type": "anomaly", "severity": "high|medium|low", "title": "short title", "description": "detailed description", "evidence_pages": [1], "evidence_text": ["source text"], "confidence": 0.0-1.0}}
  ],
  "findings": [
    {{"insight_type": "finding", "severity": "high|medium|low", "title": "short title", "description": "detailed description", "evidence_pages": [1], "evidence_text": ["source text"], "confidence": 0.0-1.0}}
  ],
  "missing_information": [
    {{"insight_type": "missing", "severity": "high|medium|low", "title": "what is missing", "description": "why it's expected", "evidence_pages": [], "evidence_text": [], "confidence": 0.0-1.0}}
  ],
  "contradictions": [
    {{"insight_type": "contradiction", "severity": "high|medium|low", "title": "short title", "description": "detailed description", "evidence_pages": [1, 2], "evidence_text": ["text1", "text2"], "confidence": 0.0-1.0}}
  ]
}}""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        parsed = safe_parse_json(response.text)
        if parsed and any(k in parsed for k in ("timeline_events", "anomalies", "findings", "missing_information", "contradictions")):
            return parsed
        raise ValueError("Invalid insights response structure")
    except Exception as e:
        logger.warning(f"Gemini insight generation failed: {e}. Using rule-based fallback.")
        return generate_insights_rule_based(text, document_type, entities, page_texts)
