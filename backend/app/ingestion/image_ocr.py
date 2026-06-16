import io
import re
import pytesseract
from PIL import Image

RAG_WORD_THRESHOLD = 3000
LOW_DPI_THRESHOLD = 150


def _load_image(file_bytes: bytes) -> tuple[Image.Image, int]:
    img = Image.open(io.BytesIO(file_bytes))
    dpi_info = img.info.get("dpi", (72, 72))
    dpi = int(dpi_info[0]) if isinstance(dpi_info, tuple) else int(dpi_info)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, dpi


def _clean_ocr_text(text: str) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _derive_name(filename: str) -> str:
    name = filename
    for ext in (".jpg", ".jpeg", ".png"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name.replace("_", " ").replace("-", " ").title()


def ocr_image(file_bytes: bytes, filename: str) -> dict:
    img, dpi = _load_image(file_bytes)
    raw_text = pytesseract.image_to_string(img, lang="eng", config="--psm 3")

    if not raw_text.strip():
        raise ValueError(
            f"{filename} produced no text from OCR. "
            "The image may be too low resolution or not contain printed text."
        )

    cleaned = _clean_ocr_text(raw_text)
    word_count = len(cleaned.split())

    return {
        "name": _derive_name(filename),
        "source_type": "image",
        "raw_text": cleaned,
        "word_count": word_count,
        "used_ocr": True,
        "will_use_rag": word_count > RAG_WORD_THRESHOLD,
        "low_dpi_warning": dpi < LOW_DPI_THRESHOLD,
    }
