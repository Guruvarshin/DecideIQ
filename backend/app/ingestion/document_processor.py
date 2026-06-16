from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.html_cleaner import clean_html
from app.ingestion.image_ocr import ocr_image

RAG_WORD_THRESHOLD = 3000

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".jpg", ".jpeg", ".png"}


def process_upload(file_bytes: bytes, filename: str) -> dict:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    if lower.endswith((".html", ".htm")):
        return clean_html(file_bytes, filename)
    if lower.endswith(".txt"):
        return process_text_paste(
            file_bytes.decode("utf-8", errors="replace"),
            filename[:-4].replace("_", " ").replace("-", " ").title(),
        )
    if lower.endswith((".jpg", ".jpeg", ".png")):
        return ocr_image(file_bytes, filename)
    raise ValueError(
        f"Unsupported file type: {filename}. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def process_text_paste(text: str, name: str) -> dict:
    cleaned = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    word_count = len(cleaned.split())
    return {
        "name": name or "Pasted Document",
        "source_type": "text",
        "raw_text": cleaned,
        "word_count": word_count,
        "will_use_rag": word_count > RAG_WORD_THRESHOLD,
    }
