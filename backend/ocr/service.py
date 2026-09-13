"""
DocuLens AI — OCR routing service
Routes document to the cheapest/most reliable processing strategy:
  - Digital PDF → pdfplumber (zero LLM cost)
  - Scanned PDF / Image → Gemini Vision OCR
"""
import io
from pathlib import Path

import pdfplumber
from PIL import Image

from backend.gemini_client import extract_text_from_image
from backend.storage.file_store import save_page_image


async def analyze_document_quality(file_path: Path) -> dict:
    """
    Determine if document is digital text, scanned, or image.
    Returns routing decision.
    """
    ext = file_path.suffix.lower()

    if ext in (".png", ".jpg", ".jpeg"):
        return {
            "is_image": True,
            "is_scanned": True,
            "has_native_text": False,
            "page_count": 1,
            "recommended_pipeline": ["vision_ocr"],
        }

    if ext == ".pdf":
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                page_count = len(pdf.pages)
                total_chars = 0
                sample_pages = min(3, page_count)

                for i in range(sample_pages):
                    text = pdf.pages[i].extract_text() or ""
                    total_chars += len(text.strip())

                has_text = total_chars > 50  # Threshold: >50 chars in first 3 pages

                return {
                    "is_image": False,
                    "is_scanned": not has_text,
                    "has_native_text": has_text,
                    "page_count": page_count,
                    "recommended_pipeline": ["pdf_parser"] if has_text else ["vision_ocr"],
                }
        except Exception:
            return {
                "is_image": False,
                "is_scanned": True,
                "has_native_text": False,
                "page_count": 0,
                "recommended_pipeline": ["vision_ocr"],
            }

    return {
        "is_image": False,
        "is_scanned": True,
        "has_native_text": False,
        "page_count": 0,
        "recommended_pipeline": ["vision_ocr"],
    }


async def extract_from_digital_pdf(file_path: Path, doc_id: str) -> dict:
    """
    Extract text and layout from a digital PDF using pdfplumber (zero LLM cost).
    Returns page data with blocks and bounding boxes.
    """
    pages_data = []

    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            width = page.width
            height = page.height

            # Extract words with bounding boxes
            words = page.extract_words() or []
            blocks = []

            # Group words into lines/blocks by vertical position
            if words:
                current_block_words = []
                current_top = None
                block_idx = 0

                for word in words:
                    if current_top is None or abs(word["top"] - current_top) > 5:
                        # New line/block
                        if current_block_words:
                            block_text = " ".join(w["text"] for w in current_block_words)
                            x0 = min(w["x0"] for w in current_block_words)
                            y0 = min(w["top"] for w in current_block_words)
                            x1 = max(w["x1"] for w in current_block_words)
                            y1 = max(w["bottom"] for w in current_block_words)

                            # Determine block type
                            block_type = _infer_block_type(block_text, current_block_words, width)

                            blocks.append({
                                "block_type": block_type,
                                "text": block_text,
                                "bbox": [
                                    x0 / width,
                                    y0 / height,
                                    x1 / width,
                                    y1 / height,
                                ],
                                "confidence": 0.98,  # Native PDF text is high confidence
                                "order_index": block_idx,
                            })
                            block_idx += 1

                        current_block_words = [word]
                        current_top = word["top"]
                    else:
                        current_block_words.append(word)

                # Last block
                if current_block_words:
                    block_text = " ".join(w["text"] for w in current_block_words)
                    x0 = min(w["x0"] for w in current_block_words)
                    y0 = min(w["top"] for w in current_block_words)
                    x1 = max(w["x1"] for w in current_block_words)
                    y1 = max(w["bottom"] for w in current_block_words)
                    block_type = _infer_block_type(block_text, current_block_words, width)
                    blocks.append({
                        "block_type": block_type,
                        "text": block_text,
                        "bbox": [x0 / width, y0 / height, x1 / width, y1 / height],
                        "confidence": 0.98,
                        "order_index": block_idx,
                    })

            # Extract tables
            tables = page.extract_tables() or []
            for tidx, table in enumerate(tables):
                if table:
                    blocks.append({
                        "block_type": "table",
                        "text": str(table),
                        "bbox": [0, 0, 1, 1],  # Tables don't have precise bbox from pdfplumber
                        "confidence": 0.95,
                        "order_index": len(blocks),
                    })

            # Render page as image for the PDF viewer
            page_image_bytes = _render_pdf_page_to_image(file_path, page_num - 1)
            if page_image_bytes:
                save_page_image(doc_id, page_num, page_image_bytes)

            pages_data.append({
                "page_number": page_num,
                "width": int(width),
                "height": int(height),
                "text": text,
                "ocr_confidence": 0.98,
                "blocks": blocks,
            })

    return {"pages": pages_data, "method": "pdf_parser"}


def _render_pdf_page_to_image(file_path: Path, page_index: int) -> bytes | None:
    """Render a PDF page to a PNG image using pypdfium2 or pdfplumber."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(file_path))
        if page_index < len(pdf):
            page = pdf[page_index]
            pil_image = page.render(scale=2.0).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            if page_index < len(pdf.pages):
                page = pdf.pages[page_index]
                img = page.to_image(resolution=150)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
    except Exception:
        pass
    return None


def _infer_block_type(text: str, words: list, page_width: float) -> str:
    """Infer if a text block is a heading, key-value pair, or regular text."""
    # Short text that looks like a heading
    if len(words) <= 8 and text.strip().endswith(":"):
        return "key_value"

    # All caps or large font (approximation: fewer words, centered)
    if text.isupper() and len(words) <= 10:
        return "heading"

    # Short centered text
    if words:
        avg_x = sum(w["x0"] for w in words) / len(words)
        if avg_x > page_width * 0.25 and len(words) <= 10:
            return "heading"

    return "text"


async def extract_from_image(file_path: Path, doc_id: str) -> dict:
    """
    Extract text from image/scanned document using Gemini Vision.
    """
    ext = file_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    image_bytes = file_path.read_bytes()

    # Save as page 1
    save_page_image(doc_id, 1, image_bytes, ext.lstrip("."))

    # Get image dimensions
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    # Call Gemini Vision for OCR
    result = await extract_text_from_image(image_bytes, mime_type)

    blocks = result.get("blocks", [])
    full_text = result.get("full_text", "")
    ocr_confidence = result.get("ocr_confidence", 0.85)

    return {
        "pages": [{
            "page_number": 1,
            "width": width,
            "height": height,
            "text": full_text,
            "ocr_confidence": ocr_confidence,
            "blocks": [
                {
                    "block_type": b.get("block_type", "text"),
                    "text": b.get("text", ""),
                    "bbox": b.get("bbox", [0, 0, 1, 1]),
                    "confidence": b.get("confidence", 0.85),
                    "order_index": i,
                }
                for i, b in enumerate(blocks)
            ],
        }],
        "method": "vision_ocr",
    }


async def extract_from_scanned_pdf(file_path: Path, doc_id: str) -> dict:
    """
    Extract text from a scanned PDF by rendering each page as image
    and sending to Gemini Vision.
    """
    pages_data = []

    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            width = int(page.width)
            height = int(page.height)

            # Render page to image
            page_image_bytes = _render_pdf_page_to_image(file_path, page_num - 1)
            if not page_image_bytes:
                continue

            save_page_image(doc_id, page_num, page_image_bytes)

            # OCR via Gemini Vision
            result = await extract_text_from_image(page_image_bytes, "image/png")

            blocks = result.get("blocks", [])
            full_text = result.get("full_text", "")
            ocr_confidence = result.get("ocr_confidence", 0.85)

            pages_data.append({
                "page_number": page_num,
                "width": width,
                "height": height,
                "text": full_text,
                "ocr_confidence": ocr_confidence,
                "blocks": [
                    {
                        "block_type": b.get("block_type", "text"),
                        "text": b.get("text", ""),
                        "bbox": b.get("bbox", [0, 0, 1, 1]),
                        "confidence": b.get("confidence", 0.85),
                        "order_index": i,
                    }
                    for i, b in enumerate(blocks)
                ],
            })

    return {"pages": pages_data, "method": "vision_ocr"}
