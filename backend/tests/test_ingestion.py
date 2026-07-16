"""
Run inside the backend container:
  docker exec decideiq-backend-1 python tests/test_ingestion.py
"""
import sys
import os
sys.path.insert(0, "/app")

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.html_cleaner import clean_html
from app.ingestion.document_processor import process_text_paste

DATA = "/app/data"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def show(result):
    rag_label = "RAG" if result["will_use_rag"] else "FULL-CTX"
    print(f"  {PASS} {result['name']}")
    print(f"     words={result['word_count']}  mode={rag_label}  type={result['source_type']}")
    print(f"     sample: {result['raw_text'][:120].replace(chr(10),' ')}")

errors = []

section("INSURANCE PDFs")
for fname in sorted(os.listdir(f"{DATA}/insurance")):
    if fname.endswith(".pdf"):
        with open(f"{DATA}/insurance/{fname}", "rb") as f:
            try:
                result = parse_pdf(f.read(), fname)
                show(result)
            except Exception as e:
                print(f"  {FAIL} {fname}: {e}")
                errors.append(fname)

section("GOLD HTMLs")
for fname in sorted(os.listdir(f"{DATA}/gold")):
    if fname.endswith(".html"):
        with open(f"{DATA}/gold/{fname}", "rb") as f:
            try:
                result = clean_html(f.read(), fname)
                show(result)
            except Exception as e:
                print(f"  {FAIL} {fname}: {e}")
                errors.append(fname)

section("SMARTPHONE HTMLs")
for fname in sorted(os.listdir(f"{DATA}/smartphone")):
    if fname.endswith(".html"):
        with open(f"{DATA}/smartphone/{fname}", "rb") as f:
            try:
                result = clean_html(f.read(), fname)
                show(result)
            except Exception as e:
                print(f"  {FAIL} {fname}: {e}")
                errors.append(fname)

section("JOB OFFER TXTs")
for fname in sorted(os.listdir(f"{DATA}/job_offers")):
    if fname.endswith(".txt"):
        with open(f"{DATA}/job_offers/{fname}", "r", encoding="utf-8") as f:
            try:
                result = process_text_paste(f.read(), fname.replace(".txt","").replace("_"," ").title())
                show(result)
            except Exception as e:
                print(f"  {FAIL} {fname}: {e}")
                errors.append(fname)

print(f"\n{'─'*60}")
if errors:
    print(f"  {FAIL} {len(errors)} file(s) failed: {errors}")
else:
    print(f"  {PASS} All documents parsed successfully")
print(f"{'─'*60}\n")
