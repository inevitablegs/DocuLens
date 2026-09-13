"""
DocuLens AI — Document classification service
Cheap-first (rules) → expensive-second (Gemini) strategy.
"""
import re
from backend.gemini_client import classify_document


# ─── Rule-based classifier ────────────────────────────────
CLASSIFICATION_RULES: dict[str, list[str]] = {
    "invoice": [
        "invoice number", "invoice no", "invoice date", "bill to", "ship to",
        "subtotal", "total amount", "tax amount", "payment terms", "due date",
        "purchase order", "unit price", "qty", "quantity",
    ],
    "research_paper": [
        "abstract", "introduction", "methodology", "references", "doi",
        "keywords", "literature review", "conclusion", "et al",
        "hypothesis", "findings", "results and discussion",
    ],
    "medical_report": [
        "patient name", "patient id", "diagnosis", "medication",
        "lab results", "blood pressure", "prescription", "dosage",
        "clinical", "symptoms", "treatment", "follow-up",
    ],
    "report": [
        "executive summary", "findings", "recommendations",
        "table of contents", "appendix", "overview", "analysis",
    ],
}


def rule_based_classify(text: str) -> tuple[str | None, float, list[str]]:
    """
    Classify using keyword rules.
    Returns (document_type, confidence, matching_signals).
    Returns None type if no clear winner.
    """
    text_lower = text.lower()
    scores: dict[str, tuple[int, list[str]]] = {}

    for doc_type, keywords in CLASSIFICATION_RULES.items():
        matches = []
        for kw in keywords:
            if kw in text_lower:
                matches.append(kw)
        if matches:
            scores[doc_type] = (len(matches), matches)

    if not scores:
        return None, 0.0, []

    # Sort by match count
    ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    best_type, (best_count, best_signals) = ranked[0]

    # Need at least 2 matches and clear winner
    if best_count < 2:
        return None, 0.0, best_signals

    # Confidence based on match count (3-5 matches indicate strong document certainty)
    confidence = min(0.95, 0.55 + min(1.0, best_count / 5.0) * 0.40)

    # Check if there's a close second
    if len(ranked) > 1:
        second_count = ranked[1][1][0]
        if second_count >= best_count - 1:
            # Too close — reduce confidence
            confidence *= 0.7

    return best_type, confidence, best_signals


async def classify(text: str) -> dict:
    """
    Classify a document using the hybrid approach:
    1. Try rule-based classification first (cheap)
    2. Fall back to Gemini if rules are inconclusive (expensive)
    """
    doc_type, confidence, signals = rule_based_classify(text)

    if doc_type and confidence >= 0.7:
        return {
            "document_type": doc_type,
            "confidence": confidence,
            "signals": signals,
            "method": "rule_based",
        }

    # Fall back to Gemini
    try:
        result = await classify_document(text)
        result["method"] = "gemini"
        return result
    except Exception as e:
        # If Gemini fails, return rule-based result or default
        if doc_type:
            return {
                "document_type": doc_type,
                "confidence": max(confidence, 0.4),
                "signals": signals,
                "method": "rule_based_fallback",
            }
        return {
            "document_type": "report",
            "confidence": 0.3,
            "signals": [],
            "method": "default_fallback",
        }
