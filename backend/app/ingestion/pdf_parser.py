import re
import fitz
import pytesseract
from PIL import Image

RAG_WORD_THRESHOLD = 3000
OCR_TRIGGER_CHARS = 50


def _clean_pdf_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n[ \t]*\d+[ \t]*\n", "\n", text)
    text = re.sub(r"\nPage \d+ of \d+\n", "\n", text, flags=re.IGNORECASE)
    return text.strip()


def _derive_name(filename: str) -> str:
    name = filename
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace("_", " ").replace("-", " ").title()


def _ocr_page(page) -> str:
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang="eng", config="--psm 3")


def parse_pdf(file_bytes: bytes, filename: str) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)
    pages = []
    ocr_page_count = 0

    for page in doc:
        text = page.get_text("text")
        if len(text.strip()) >= OCR_TRIGGER_CHARS:
            pages.append(text)
        else:
            ocr_text = _ocr_page(page)
            if ocr_text.strip():
                pages.append(ocr_text)
                ocr_page_count += 1

    doc.close()

    if not pages:
        raise ValueError(
            f"{filename} has no extractable text and OCR found nothing. "
            "The document may be too low quality or in an unsupported language."
        )

    raw_text = "\n\n".join(pages)
    cleaned = _clean_pdf_text(raw_text)
    word_count = len(cleaned.split())

    return {
        "name": _derive_name(filename),
        "source_type": "pdf",
        "raw_text": cleaned,
        "word_count": word_count,
        "page_count": page_count,
        "ocr_page_count": ocr_page_count,
        "used_ocr": ocr_page_count > 0,
        "will_use_rag": word_count > RAG_WORD_THRESHOLD,
    }
