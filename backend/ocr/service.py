"""
DocuLens AI — OCR and Document Parsing Service
Routes document to the cheapest/most reliable processing strategy:
  - Digital PDF → pdfplumber (zero LLM cost)
  - Scanned PDF / Image → Gemini Vision OCR
  - Resilient fallbacks for text-only mock PDFs, malformed files, or missing images
"""
import io
import logging
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

from backend.gemini_client import extract_text_from_image
from backend.storage.file_store import save_page_image

logger = logging.getLogger(__name__)


def _generate_text_canvas_image(text: str, page_num: int = 1, width: int = 800, height: int = 1050) -> bytes:
    """
    Generate an aesthetic document canvas preview when native rendering is unavailable.
    Creates a crisp document page representation for the frontend viewer.
    """
    img = Image.new("RGB", (width, height), color="#0e1322")
    draw = ImageDraw.Draw(img)

    # Document border/card effect
    draw.rectangle([20, 20, width - 20, height - 20], fill="#141b2d", outline="#2a3550", width=2)
    draw.rectangle([20, 20, width - 20, 70], fill="#1c253d")

    # Header bar
    draw.text((40, 35), f"DOCULENS DOCUMENT VIEWER · PAGE {page_num}", fill="#818cf8")

    # Render lines
    lines = text.split("\n")
    y = 100
    line_height = 24

    for line in lines:
        if y > height - 60:
            draw.text((40, y), "... [Remaining content truncated for preview]", fill="#64748b")
            break
        stripped = line.strip()
        if not stripped:
            y += 12
            continue

        # Color key words or headings
        if any(keyword in stripped.upper() for keyword in ("INVOICE", "TOTAL", "REPORT", "SUMMARY", "PATIENT", "DIAGNOSIS")):
            draw.text((40, y), stripped[:90], fill="#38bdf8")
        else:
            draw.text((40, y), stripped[:90], fill="#cbd5e1")
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _read_raw_text_safe(file_path: Path) -> str:
    """Read readable text from a file, handling UTF-8, Latin-1, or binary stripping."""
    try:
        raw = file_path.read_bytes()
    except Exception:
        return ""

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            decoded = raw.decode(enc)
            # Check if it has readable text
            printable = "".join(c for c in decoded if c.isprintable() or c in "\n\r\t")
            if len(printable.strip()) >= 5:
                return printable
        except Exception:
            continue

    return ""


def _extract_text_fallback(file_path: Path, doc_id: str) -> dict:
    """Fallback extraction for mock PDFs or files that pdfplumber/pdfium cannot parse."""
    raw_text = _read_raw_text_safe(file_path)
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    if not lines:
        lines = ["Document content could not be rendered."]
        raw_text = lines[0]

    width = 800
    height = 1050
    total_lines = len(lines)
    blocks = []

    for idx, line in enumerate(lines):
        y0_norm = (idx / max(total_lines, 1)) * 0.8 + 0.08
        y1_norm = min(1.0, y0_norm + (1.0 / max(total_lines, 1)) * 0.8)
        block_type = "heading" if (idx == 0 or line.isupper() or len(line.split()) <= 4) else "text"
        if ":" in line and len(line.split()) <= 8:
            block_type = "key_value"

        blocks.append({
            "block_type": block_type,
            "text": line,
            "bbox": [0.08, y0_norm, 0.92, y1_norm],
            "confidence": 0.95,
            "order_index": idx,
        })

    # Generate preview image
    img_bytes = _generate_text_canvas_image(raw_text, page_num=1, width=width, height=height)
    save_page_image(doc_id, 1, img_bytes, "png")

    return {
        "pages": [{
            "page_number": 1,
            "width": width,
            "height": height,
            "text": raw_text,
            "ocr_confidence": 0.95,
            "blocks": blocks,
        }],
        "method": "text_fallback",
    }


async def analyze_document_quality(file_path: Path) -> dict:
    """
    Determine if document is digital text, scanned, or image.
    Returns routing decision safely without crashing on corrupted or mock files.
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

                has_text = total_chars > 30

                return {
                    "is_image": False,
                    "is_scanned": not has_text,
                    "has_native_text": has_text,
                    "page_count": max(1, page_count),
                    "recommended_pipeline": ["pdf_parser"] if has_text else ["vision_ocr"],
                }
        except Exception as e:
            logger.info(f"pdfplumber inspect failed on {file_path.name}: {e}. Checking text fallback.")
            # Check if file has readable text (e.g. mock PDF)
            raw = _read_raw_text_safe(file_path)
            if len(raw.strip()) > 15:
                return {
                    "is_image": False,
                    "is_scanned": False,
                    "has_native_text": True,
                    "page_count": 1,
                    "recommended_pipeline": ["pdf_parser"],
                }
            return {
                "is_image": False,
                "is_scanned": True,
                "has_native_text": False,
                "page_count": 1,
                "recommended_pipeline": ["vision_ocr"],
            }

    return {
        "is_image": False,
        "is_scanned": True,
        "has_native_text": False,
        "page_count": 1,
        "recommended_pipeline": ["vision_ocr"],
    }


async def extract_from_digital_pdf(file_path: Path, doc_id: str) -> dict:
    """
    Extract text and layout from a digital PDF using pdfplumber (zero LLM cost).
    Falls back gracefully if PDF format is invalid.
    """
    pages_data = []

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            if not pdf.pages:
                return _extract_text_fallback(file_path, doc_id)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                width = float(page.width or 612)
                height = float(page.height or 792)

                words = page.extract_words() or []
                blocks = []

                if words:
                    current_block_words = []
                    current_top = None
                    block_idx = 0

                    for word in words:
                        w_top = word.get("top", 0)
                        if current_top is None or abs(w_top - current_top) > 5:
                            if current_block_words:
                                b_text = " ".join(w.get("text", "") for w in current_block_words)
                                x0 = min(w.get("x0", 0) for w in current_block_words)
                                y0 = min(w.get("top", 0) for w in current_block_words)
                                x1 = max(w.get("x1", 1) for w in current_block_words)
                                y1 = max(w.get("bottom", 1) for w in current_block_words)
                                b_type = _infer_block_type(b_text, current_block_words, width)

                                blocks.append({
                                    "block_type": b_type,
                                    "text": b_text,
                                    "bbox": [
                                        max(0.0, min(1.0, x0 / max(width, 1))),
                                        max(0.0, min(1.0, y0 / max(height, 1))),
                                        max(0.0, min(1.0, x1 / max(width, 1))),
                                        max(0.0, min(1.0, y1 / max(height, 1))),
                                    ],
                                    "confidence": 0.98,
                                    "order_index": block_idx,
                                })
                                block_idx += 1

                            current_block_words = [word]
                            current_top = w_top
                        else:
                            current_block_words.append(word)

                    if current_block_words:
                        b_text = " ".join(w.get("text", "") for w in current_block_words)
                        x0 = min(w.get("x0", 0) for w in current_block_words)
                        y0 = min(w.get("top", 0) for w in current_block_words)
                        x1 = max(w.get("x1", 1) for w in current_block_words)
                        y1 = max(w.get("bottom", 1) for w in current_block_words)
                        b_type = _infer_block_type(b_text, current_block_words, width)
                        blocks.append({
                            "block_type": b_type,
                            "text": b_text,
                            "bbox": [
                                max(0.0, min(1.0, x0 / max(width, 1))),
                                max(0.0, min(1.0, y0 / max(height, 1))),
                                max(0.0, min(1.0, x1 / max(width, 1))),
                                max(0.0, min(1.0, y1 / max(height, 1))),
                            ],
                            "confidence": 0.98,
                            "order_index": block_idx,
                        })

                # If no word-level bounding boxes were parsed, create lines as blocks
                if not blocks and text:
                    for idx, line in enumerate(text.split("\n")):
                        if line.strip():
                            blocks.append({
                                "block_type": "text",
                                "text": line.strip(),
                                "bbox": [0.08, (idx / 40.0) % 1.0, 0.92, ((idx + 1) / 40.0) % 1.0],
                                "confidence": 0.95,
                                "order_index": idx,
                            })

                # Extract tables
                try:
                    tables = page.extract_tables() or []
                    for table in tables:
                        if table:
                            blocks.append({
                                "block_type": "table",
                                "text": str(table),
                                "bbox": [0.05, 0.05, 0.95, 0.95],
                                "confidence": 0.95,
                                "order_index": len(blocks),
                            })
                except Exception:
                    pass

                # Render page as image for the PDF viewer
                page_image_bytes = _render_pdf_page_to_image(file_path, page_num - 1)
                if not page_image_bytes:
                    page_image_bytes = _generate_text_canvas_image(text, page_num=page_num, width=int(width), height=int(height))

                save_page_image(doc_id, page_num, page_image_bytes, "png")

                pages_data.append({
                    "page_number": page_num,
                    "width": int(width),
                    "height": int(height),
                    "text": text,
                    "ocr_confidence": 0.98,
                    "blocks": blocks,
                })

        return {"pages": pages_data, "method": "pdf_parser"}
    except Exception as e:
        logger.warning(f"extract_from_digital_pdf failed on {file_path}: {e}. Using fallback.")
        return _extract_text_fallback(file_path, doc_id)


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
                if hasattr(img, "original"):
                    img.original.save(buf, format="PNG")
                elif hasattr(img, "save"):
                    img.save(buf, format="PNG")
                return buf.getvalue()
    except Exception:
        pass
    return None


def _infer_block_type(text: str, words: list, page_width: float) -> str:
    """Infer if a text block is a heading, key-value pair, or regular text."""
    if len(words) <= 8 and text.strip().endswith(":"):
        return "key_value"

    if text.isupper() and len(words) <= 10:
        return "heading"

    if words:
        avg_x = sum(w.get("x0", 0) for w in words) / max(len(words), 1)
        if avg_x > page_width * 0.25 and len(words) <= 10:
            return "heading"

    return "text"


async def extract_from_image(file_path: Path, doc_id: str) -> dict:
    """Extract text from image/scanned document using Gemini Vision or fallback."""
    ext = file_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    image_bytes = file_path.read_bytes()

    # Save image for viewer
    save_page_image(doc_id, 1, image_bytes, ext.lstrip("."))

    # Get image dimensions safely
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
    except Exception:
        width, height = 800, 1000

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
                    "bbox": b.get("bbox", [0.05, 0.05, 0.95, 0.95]),
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
    and sending to Gemini Vision. Falls back safely if PDF cannot be opened.
    """
    pages_data = []

    try:
        page_count = 0
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                page_count = len(pdf.pages)
        except Exception:
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(file_path))
                page_count = len(pdf)
            except Exception:
                page_count = 0

        if page_count == 0:
            return _extract_text_fallback(file_path, doc_id)

        for page_idx in range(page_count):
            page_num = page_idx + 1
            page_image_bytes = _render_pdf_page_to_image(file_path, page_idx)
            if not page_image_bytes:
                # Generate fallback image canvas
                page_image_bytes = _generate_text_canvas_image(f"Page {page_num}", page_num=page_num)

            save_page_image(doc_id, page_num, page_image_bytes, "png")

            # OCR via Gemini Vision
            result = await extract_text_from_image(page_image_bytes, "image/png")

            blocks = result.get("blocks", [])
            full_text = result.get("full_text", "")
            ocr_confidence = result.get("ocr_confidence", 0.85)

            pages_data.append({
                "page_number": page_num,
                "width": 800,
                "height": 1050,
                "text": full_text,
                "ocr_confidence": ocr_confidence,
                "blocks": [
                    {
                        "block_type": b.get("block_type", "text"),
                        "text": b.get("text", ""),
                        "bbox": b.get("bbox", [0.05, 0.05, 0.95, 0.95]),
                        "confidence": b.get("confidence", 0.85),
                        "order_index": i,
                    }
                    for i, b in enumerate(blocks)
                ],
            })

        return {"pages": pages_data, "method": "vision_ocr"}
    except Exception as e:
        logger.warning(f"extract_from_scanned_pdf failed on {file_path}: {e}. Using fallback.")
        return _extract_text_fallback(file_path, doc_id)
