"""
DocuLens AI — Confidence scoring service
Composite confidence — NOT LLM-invented numbers.
"""
from backend.config import CONFIDENCE_WEIGHTS, CONFIDENCE_THRESHOLD


def compute_entity_confidence(
    ocr_confidence: float = 1.0,
    extraction_confidence: float = 0.9,
    schema_valid: bool = True,
    evidence_match_score: float = 0.9,
    model_confidence: float = 0.9,
) -> float:
    """
    Compute composite confidence for a single entity.

    Formula:
    Final = 0.30 × OCR + 0.25 × extraction + 0.20 × schema + 0.15 × evidence + 0.10 × model
    """
    schema_score = 1.0 if schema_valid else 0.5

    weighted = (
        CONFIDENCE_WEIGHTS["ocr"] * ocr_confidence
        + CONFIDENCE_WEIGHTS["extraction"] * extraction_confidence
        + CONFIDENCE_WEIGHTS["schema"] * schema_score
        + CONFIDENCE_WEIGHTS["evidence"] * evidence_match_score
        + CONFIDENCE_WEIGHTS["model"] * model_confidence
    )

    return round(min(1.0, max(0.0, weighted)), 4)


def compute_document_confidence(entity_confidences: list[float]) -> float:
    """Compute overall document confidence as weighted average of entity confidences."""
    if not entity_confidences:
        return 0.0
    return round(sum(entity_confidences) / len(entity_confidences), 4)


def needs_human_review(confidence: float) -> bool:
    """Check if an item's confidence is below threshold for human review."""
    return confidence < CONFIDENCE_THRESHOLD


def compute_evidence_match_score(entity_value: str, source_text: str) -> float:
    """
    Compute how well an entity value matches its source text.
    Uses fuzzy string matching.
    """
    if not entity_value or not source_text:
        return 0.0

    entity_lower = entity_value.lower().strip()
    source_lower = source_text.lower().strip()

    # Exact containment
    if entity_lower in source_lower:
        return 1.0

    # Partial match — character overlap
    entity_chars = set(entity_lower)
    source_chars = set(source_lower)
    if not entity_chars:
        return 0.0

    overlap = len(entity_chars & source_chars) / len(entity_chars)

    # Check for word-level matches
    entity_words = set(entity_lower.split())
    source_words = set(source_lower.split())
    if entity_words:
        word_overlap = len(entity_words & source_words) / len(entity_words)
        return round(max(overlap * 0.6, word_overlap), 4)

    return round(overlap * 0.6, 4)
