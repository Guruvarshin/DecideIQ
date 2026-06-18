FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FlashRank model so it's baked into the image (avoids corrupt mid-run downloads).
# Retry up to 3 times; if all fail the build still succeeds and the model downloads at first run.
RUN for i in 1 2 3; do \
      python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/models/flashrank')" && break; \
      echo "FlashRank download attempt $i failed, retrying..."; \
      sleep 5; \
    done; exit 0

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--loop", "asyncio"]
