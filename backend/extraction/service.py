"""
DocuLens AI — Entity and Table Extraction Service
Provides schema-guided rule/regex extraction with block-level bounding box resolution,
serving both as standalone extraction and as an intelligent offline fallback for LLMs.
"""
import re
from typing import Any


# ─── Regex patterns per document type ────────────────────
INVOICE_PATTERNS = {
    "INVOICE_NUMBER": [
        r"(?:invoice\s*(?:no|number|#|id)[\s:]*)([A-Z0-9\-_/]+)",
        r"(?:inv[\s#:]+)([A-Z0-9\-_/]+)",
    ],
    "INVOICE_DATE": [
        r"(?:invoice\s*date[\s:]*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        r"(?:date[\s:]*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    ],
    "DUE_DATE": [
        r"(?:due\s*date[\s:]*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        r"(?:payment\s*due[\s:]*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    ],
    "TOTAL_AMOUNT": [
        r"(?:total\s*(?:amount|due|balance)?[\s:]*)(?:[$€£₹]\s*|\bUSD\s*|\bEUR\s*)?([0-9,]+\.\d{2})",
        r"(?:amount\s*due[\s:]*)(?:[$€£₹]\s*)?([0-9,]+\.\d{2})",
        r"(?:balance\s*due[\s:]*)(?:[$€£₹]\s*)?([0-9,]+\.\d{2})",
    ],
    "SUBTOTAL": [
        r"(?:subtotal|sub\s*total|net\s*amount)[\s:]*(?:[$€£₹]\s*)?([0-9,]+\.\d{2})",
    ],
    "TAX": [
        r"(?:tax|vat|gst|sales\s*tax)[\s:]*(?:[$€£₹]\s*)?([0-9,]+\.\d{2})",
    ],
    "VENDOR": [
        r"(?:from|vendor|billed\s*by|remit\s*to)[\s:]*([A-Za-z0-9\s.,&'-]{3,40})",
        r"^([A-Z][A-Za-z0-9\s.,&'-]{2,30}(?:Inc\.?|LLC|Corp\.?|Ltd\.?|GmbH|Co\.?))",
    ],
    "BILLING_ADDRESS": [
        r"(?:bill\s*to|sold\s*to|client)[\s:]*([A-Za-z0-9\s.,#\n-]{10,80})",
    ],
    "PAYMENT_TERMS": [
        r"(?:payment\s*terms|terms)[\s:]*([A-Za-z0-9\s]+(?:net\s*\d+|due\s*upon\s*receipt|receipt))",
        r"\b(net\s*\d+)\b",
    ],
}

MEDICAL_PATTERNS = {
    "PATIENT_NAME": [
        r"(?:patient\s*(?:name)?[\s:]*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"(?:name[\s:]*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ],
    "PATIENT_ID": [
        r"(?:patient\s*id|mrn|medical\s*record\s*#?)[\s:]*([A-Z0-9\-_]+)",
        r"(?:id[\s:]*)([A-Z0-9\-_]{5,15})",
    ],
    "DATE": [
        r"(?:date\s*(?:of\s*service|of\s*visit)?[\s:]*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    ],
    "DOCTOR": [
        r"(?:attending\s*physician|physician|doctor|dr\.)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ],
    "DEPARTMENT": [
        r"(?:department|dept|clinic)[\s:]*([A-Za-z\s]{4,35})",
    ],
    "DIAGNOSIS": [
        r"(?:diagnosis|primary\s*diagnosis|impression|assessment)[\s:]*([A-Za-z0-9\s.,-]{4,60})",
    ],
    "MEDICATION": [
        r"(?:rx|medication|prescribed)[\s:]*([A-Za-z0-9\s.,-]{3,40})",
    ],
    "DOSAGE": [
        r"(\d+\s*(?:mg|ml|mcg|tablets?|capsules?)\s*(?:daily|bid|tid|qid|once\s*daily|as\s*needed)?)",
    ],
    "VITAL_SIGN": [
        r"(?:bp|blood\s*pressure)[\s:]*(\d{2,3}/\d{2,3})",
        r"(?:heart\s*rate|pulse)[\s:]*(\d{2,3}\s*(?:bpm)?)",
    ],
}

RESEARCH_PATTERNS = {
    "TITLE": [
        r"^([A-Z][A-Za-z0-9\s:,-]{10,120})(?=\n|$)",
        r"(?:title[\s:]*)([A-Za-z0-9\s:,-]{10,120})",
    ],
    "AUTHOR": [
        r"(?:authors?|by)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)",
    ],
    "DOI": [
        r"(?:doi[\s:]*|(?:https?://)?doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
    ],
    "DATE": [
        r"(?:published|date|accepted)[\s:]*(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{4})",
    ],
    "ABSTRACT_SUMMARY": [
        r"(?:abstract[\s:]*)([A-Za-z0-9\s.,;:()\-'\"]{40,300})",
    ],
    "METHODOLOGY": [
        r"(?:methodology|methods|approach)[\s:]*([A-Za-z0-9\s.,;:()\-'\"]{20,200})",
    ],
}

REPORT_PATTERNS = {
    "TITLE": [
        r"^([A-Z][A-Za-z0-9\s:,-]{10,100})(?=\n|$)",
        r"(?:report\s*title|subject)[\s:]*([A-Za-z0-9\s:,-]{10,100})",
    ],
    "DATE": [
        r"(?:date|created|reported)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    ],
    "ORGANIZATION": [
        r"(?:organization|company|department|agency)[\s:]*([A-Za-z0-9\s.,&'-]{3,40})",
    ],
    "AUTHOR": [
        r"(?:prepared\s*by|author|reporter)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ],
    "KEY_FINDING": [
        r"(?:key\s*findings?|summary|conclusion)[\s:]*([A-Za-z0-9\s.,;:()\-'\"]{20,200})",
    ],
}


def _get_patterns_for_type(doc_type: str) -> dict[str, list[str]]:
    if doc_type == "invoice":
        return INVOICE_PATTERNS
    elif doc_type == "medical_report":
        return MEDICAL_PATTERNS
    elif doc_type == "research_paper":
        return RESEARCH_PATTERNS
    elif doc_type == "report":
        return REPORT_PATTERNS
    else:
        # General / combined
        combined = dict(REPORT_PATTERNS)
        combined.update(INVOICE_PATTERNS)
        return combined


def extract_entities_rule_based(
    text: str,
    doc_type: str,
    page_texts: dict[int, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract structured entities using type-tailored regex and heuristics.
    Tracks page provenance and source text for each field.
    """
    patterns = _get_patterns_for_type(doc_type)
    extracted: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    pages = page_texts if page_texts else {1: text}

    for page_num, page_str in sorted(pages.items()):
        for entity_type, regex_list in patterns.items():
            for pat in regex_list:
                matches = re.finditer(pat, page_str, re.IGNORECASE | re.MULTILINE)
                for m in matches:
                    val = m.group(1).strip()
                    # Clean up trailing punctuation
                    val = re.sub(r"[\s,:;]+$", "", val)
                    if not val or len(val) < 2:
                        continue

                    key = (entity_type, val.lower())
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    # Extract context snippet around match
                    start = max(0, m.start() - 25)
                    end = min(len(page_str), m.end() + 25)
                    snippet = page_str[start:end].replace("\n", " ").strip()

                    extracted.append({
                        "entity_type": entity_type,
                        "value": val,
                        "page": page_num,
                        "source_text": snippet,
                        "confidence": 0.88,
                    })

    return {"entities": extracted}


def extract_tables_rule_based(
    text: str,
    doc_type: str,
    page_texts: dict[int, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Heuristically detect tabular data in text (e.g. pipe-delimited, aligned columns, CSV-like).
    """
    tables: list[dict[str, Any]] = []
    pages = page_texts if page_texts else {1: text}

    for page_num, page_str in sorted(pages.items()):
        lines = [line.strip() for line in page_str.split("\n") if line.strip()]
        candidate_rows: list[list[str]] = []
        in_table = False

        for line in lines:
            # Check for pipe-separated table
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= 2 and not all(set(c).issubset({"-", "=", ":"}) for c in cells):
                    candidate_rows.append(cells)
                    in_table = True
                    continue

            # Check for multiple tab or multi-space separated columns
            parts = re.split(r"\t+|\s{3,}", line)
            if len(parts) >= 3:
                candidate_rows.append(parts)
                in_table = True
                continue

            if in_table and len(candidate_rows) >= 2:
                # Table ended
                header = candidate_rows[0]
                body = candidate_rows[1:]
                tables.append({
                    "title": f"Table (Page {page_num})",
                    "page": page_num,
                    "columns": header,
                    "rows": body,
                    "confidence": 0.85,
                })
                candidate_rows = []
                in_table = False

        if candidate_rows and len(candidate_rows) >= 2:
            tables.append({
                "title": f"Table (Page {page_num})",
                "page": page_num,
                "columns": candidate_rows[0],
                "rows": candidate_rows[1:],
                "confidence": 0.85,
            })

    return {"tables": tables}
