import re
from bs4 import BeautifulSoup

RAG_WORD_THRESHOLD = 3000

_REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "iframe", "noscript", "svg", "form",
]

_REMOVE_CLASS_FRAGMENTS = [
    "nav", "menu", "footer", "header", "sidebar", "cookie",
    "popup", "modal", "breadcrumb", "pagination", "social",
    "share", "newsletter", "subscribe", "banner", "advertisement",
    "promo", "related", "widget",
]


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup(_REMOVE_TAGS):
        tag.decompose()
    for fragment in _REMOVE_CLASS_FRAGMENTS:
        for tag in soup.find_all(
            class_=lambda c: c and any(fragment in cls.lower() for cls in c)
        ):
            tag.decompose()


def _extract_title(soup: BeautifulSoup, filename: str) -> str:
    title_tag = soup.find("title")
    if title_tag:
        raw = title_tag.get_text().strip()
        name = re.split(r"\s*[|\-–—]\s*", raw)[0].strip()
        if name:
            return name[:120]
    h1 = soup.find("h1")
    if h1:
        return h1.get_text().strip()[:120]
    return filename.replace(".html", "").replace("_", " ").title()


def _extract_content(soup: BeautifulSoup) -> str:
    for selector in ["main", "article", '[role="main"]', "#content", ".content", "body"]:
        el = soup.select_one(selector)
        if el:
            return el.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 1]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_html(file_bytes: bytes, filename: str) -> dict:
    soup = BeautifulSoup(file_bytes, "lxml")
    title = _extract_title(soup, filename)
    _strip_boilerplate(soup)
    raw_text = _extract_content(soup)
    cleaned = _clean_text(raw_text)
    word_count = len(cleaned.split())

    return {
        "name": title,
        "source_type": "html",
        "raw_text": cleaned,
        "word_count": word_count,
        "will_use_rag": word_count > RAG_WORD_THRESHOLD,
    }
