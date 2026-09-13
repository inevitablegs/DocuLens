"""
DocuLens AI — Insight Generation Service
Generates evidence-grounded insights: timeline events, anomalies, key findings,
missing information, and contradictions.
"""
import re
from typing import Any


def _parse_amount(val: str) -> float | None:
    try:
        cleaned = re.sub(r"[^\d.]", "", val)
        return float(cleaned)
    except Exception:
        return None


def generate_insights_rule_based(
    text: str,
    doc_type: str,
    entities: list[dict[str, Any]],
    page_texts: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Generate structured, evidence-grounded insights using rule-based reasoning,
    math checks, and schema expectations.
    """
    pages = page_texts if page_texts else {1: text}
    entity_map: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        t = e.get("entity_type") or e.get("type", "")
        if t not in entity_map:
            entity_map[t] = []
        entity_map[t].append(e)

    timeline_events: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    missing_information: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []

    # ──── 1. Timeline Events ─────────────────────────────────
    date_regex = r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b"
    seen_events: set[str] = set()

    for page_num, p_text in sorted(pages.items()):
        lines = p_text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str or len(line_str) < 10:
                continue
            date_match = re.search(date_regex, line_str, re.IGNORECASE)
            if date_match:
                d_str = date_match.group(1)
                event_desc = line_str[:120]
                key = f"{d_str}:{event_desc}"
                if key not in seen_events:
                    seen_events.add(key)
                    timeline_events.append({
                        "date": d_str,
                        "event": event_desc,
                        "page": page_num,
                        "confidence": 0.90,
                    })

    # ──── 2. Anomalies ──────────────────────────────────────
    if doc_type == "invoice":
        subtotal_ent = entity_map.get("SUBTOTAL", [])
        tax_ent = entity_map.get("TAX", [])
        total_ent = entity_map.get("TOTAL_AMOUNT", [])

        if subtotal_ent and total_ent:
            sub = _parse_amount(subtotal_ent[0].get("value", ""))
            tot = _parse_amount(total_ent[0].get("value", ""))
            tax = _parse_amount(tax_ent[0].get("value", "")) if tax_ent else 0.0

            if sub is not None and tot is not None and tax is not None:
                expected = round(sub + tax, 2)
                diff = abs(expected - tot)
                if diff > 0.05:
                    anomalies.append({
                        "insight_type": "anomaly",
                        "severity": "high",
                        "title": "Subtotal & Tax Calculation Mismatch",
                        "description": (
                            f"Calculated sum of subtotal ({sub:.2f}) + tax ({tax:.2f}) is {expected:.2f}, "
                            f"which differs from stated total of {tot:.2f} by {diff:.2f}."
                        ),
                        "evidence_pages": [total_ent[0].get("page", 1)],
                        "evidence_text": [total_ent[0].get("source_text", "")],
                        "confidence": 0.95,
                    })

        # Check invoice date vs due date
        inv_dates = entity_map.get("INVOICE_DATE", [])
        due_dates = entity_map.get("DUE_DATE", [])
        if inv_dates and due_dates:
            # Check string equality as suspicious immediate due date or sanity check
            pass

    # ──── 3. Contradictions ──────────────────────────────────
    # Check for multiple differing values for single-value fields
    single_value_fields = ["TOTAL_AMOUNT", "INVOICE_NUMBER", "PATIENT_ID", "DOI"]
    for field in single_value_fields:
        items = entity_map.get(field, [])
        distinct_vals = list({i.get("value", "").strip() for i in items if i.get("value")})
        if len(distinct_vals) > 1:
            contradictions.append({
                "insight_type": "contradiction",
                "severity": "medium",
                "title": f"Conflicting {field} Values Detected",
                "description": f"Found multiple conflicting values for {field}: {', '.join(distinct_vals)} across the document.",
                "evidence_pages": [i.get("page", 1) for i in items[:3]],
                "evidence_text": [i.get("source_text", "") for i in items[:3] if i.get("source_text")],
                "confidence": 0.85,
            })

    # ──── 4. Missing Information ─────────────────────────────
    expected_fields_by_type = {
        "invoice": ["INVOICE_NUMBER", "INVOICE_DATE", "TOTAL_AMOUNT", "VENDOR", "DUE_DATE", "TAX"],
        "medical_report": ["PATIENT_NAME", "PATIENT_ID", "DATE", "DOCTOR", "DIAGNOSIS"],
        "research_paper": ["TITLE", "AUTHOR", "DATE", "ABSTRACT_SUMMARY"],
        "report": ["TITLE", "DATE", "ORGANIZATION"],
    }

    expected = expected_fields_by_type.get(doc_type, ["TITLE", "DATE"])
    for field in expected:
        if field not in entity_map:
            missing_information.append({
                "insight_type": "missing",
                "severity": "medium" if field in ("DUE_DATE", "TAX") else "high",
                "title": f"Missing Expected Field: {field}",
                "description": f"Standard {doc_type} documents typically contain '{field}', but no matching entry was detected.",
                "evidence_pages": [],
                "evidence_text": [],
                "confidence": 0.90,
            })

    # ──── 5. Key Findings ────────────────────────────────────
    if doc_type == "invoice":
        vendor = entity_map.get("VENDOR", [{}])[0].get("value", "Unknown Vendor")
        total = entity_map.get("TOTAL_AMOUNT", [{}])[0].get("value", "N/A")
        due = entity_map.get("DUE_DATE", [{}])[0].get("value", "unspecified date")
        findings.append({
            "insight_type": "finding",
            "severity": "low",
            "title": f"Invoice Summary: {vendor}",
            "description": f"Invoice issued by {vendor} for total amount {total}, due on {due}.",
            "evidence_pages": [1],
            "evidence_text": [f"Vendor: {vendor}, Total: {total}"],
            "confidence": 0.92,
        })
    elif doc_type == "medical_report":
        diag = entity_map.get("DIAGNOSIS", [{}])[0].get("value")
        if diag:
            findings.append({
                "insight_type": "finding",
                "severity": "high",
                "title": f"Primary Clinical Impression: {diag}",
                "description": f"Document records primary diagnosis/impression as '{diag}'.",
                "evidence_pages": [1],
                "evidence_text": [diag],
                "confidence": 0.92,
            })
    elif doc_type == "research_paper":
        title = entity_map.get("TITLE", [{}])[0].get("value")
        if title:
            findings.append({
                "insight_type": "finding",
                "severity": "low",
                "title": "Research Paper Identified",
                "description": f"Analyzed research paper titled '{title}'.",
                "evidence_pages": [1],
                "evidence_text": [title],
                "confidence": 0.95,
            })

    return {
        "timeline_events": timeline_events[:15],
        "anomalies": anomalies,
        "findings": findings,
        "missing_information": missing_information,
        "contradictions": contradictions,
    }
