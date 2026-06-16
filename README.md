# DecideIQ - AI Decision Engine

**Live:** [decide-iq.vercel.app](https://decide-iq.vercel.app) &nbsp;|&nbsp; **API:** [decideiq.onrender.com](https://decideiq.onrender.com)

Upload your options. Ask your questions. Get one clear winner.

DecideIQ is a full-stack AI application that compares multiple documents (job offers, insurance plans, product specs, contracts) side-by-side using a production-grade RAG pipeline, and delivers a scored verdict with evidence from the source documents.

---

## What it does

1. **Upload** 2+ documents (PDF, HTML, TXT, image)
2. **Ask** what matters to you - or let the AI generate questions from your comparison title
3. **Compare** - the RAG pipeline retrieves context per document and answers every question
4. **Score** - answers are scored 1–10 comparatively; a final AI verdict names the winner
5. **Evaluate** - RAGAS metrics (faithfulness, answer relevancy) measure answer quality per document

---

## Tech stack

### Backend
| Layer | Technology |
|---|---|
| API | FastAPI (async) + Uvicorn |
| Auth | JWT (PyJWT HS256) in httpOnly cookies, bcrypt |
| Database | MongoDB Atlas via Motor (async) |
| Vector store | ChromaDB (persistent, one collection per doc) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM - answers & scoring | OpenAI `gpt-4o-mini` |
| LLM - verdict | Anthropic `claude-sonnet-4-6` |
| RAG pipeline | LangGraph stateful graph |
| Retrieval | BM25 + dense hybrid, RRF (k=60) |
| Reranking | FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) |
| Compression | LangChain contextual compression |
| Web fallback | CRAG - Tavily search with grounding check |
| Evaluation | RAGAS 0.2.14 (faithfulness + answer relevancy) |
| Ingestion | PyMuPDF, BeautifulSoup, Tesseract OCR, Pillow |

### Frontend
| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS + `@tailwindcss/typography` |
| Markdown | `react-markdown` |
| Auth | Cookie-based (`credentials: include` on all fetches) |

### Infrastructure
| Component | Technology |
|---|---|
| Containerisation | Docker + Docker Compose |
| Backend deployment | Render (Docker) |
| Frontend deployment | Vercel |

---

## RAG pipeline architecture

```
User question
    │
    ├─ Multi-query decomposition (3 sub-queries)
    ├─ HyDE (Hypothetical Document Embedding)
    │
    ▼
Hybrid retrieval
    ├─ BM25 (keyword)
    └─ Dense (ChromaDB cosine similarity)
         │
         ▼ Reciprocal Rank Fusion (k=60)
         │
         ▼ Parent chunk expansion (child→parent lookup)
         │
         ▼ FlashRank cross-encoder reranking (top 8)
         │
         ▼ Contextual compression (GPT-4o-mini)
         │
         ▼ Grounding check (cosine similarity ≥ 0.35)
              │
    ┌─────────┴──────────┐
  Grounded           Not grounded
    │                    │
  Use RAG          CRAG web search (Tavily)
  contexts         + grounding check on results
                        │
                   Grounded? → use web results
                   No?       → "Not mentioned in available sources."
```

### Chunking strategy
- **Parent chunks:** 1800 characters (full context)
- **Child chunks:** 400 characters (precise retrieval)
- Retrieval hits children → expands to parents for answering

### Question generation (LangGraph - 3 nodes)
1. **Node 1:** Rephrases user questions to be specific and measurable
2. **Node 2:** Generates 5 questions per document from its actual content (concurrently), covering key decision dimensions — 15 questions total for 3 docs
3. **Node 3:** Deduplicates across all per-doc questions, keeping the clearest phrasing of each unique question

Questions are **grounded in document content**, so the pipeline extracts what actually matters from each option rather than guessing from the title alone.

---

## RAGAS evaluation scores (golden dataset - job offer comparison)

8 factual questions evaluated against hand-written golden answers grounded in actual document text.

| Metric | TechCorp Offer | FinEdge Offer |
|---|---|---|
| Faithfulness | 0.9196 | 0.8438 |
| Answer Relevancy | 0.7042 | 0.7196 |
| Context Recall | 0.7396 | 0.6771 |
| Answer Correctness | 0.6115 | 0.5930 |
| **Confidence Score** | **81.2%** | **78.2%** |

- **Faithfulness 0.92/0.84** — very low hallucination; answers stay grounded in retrieved context
- **Context Recall 0.74/0.68** — RAG retrieves most relevant passages; misses on questions where content spans multiple scattered sections
- **Answer Correctness ~0.60** — pipeline sometimes paraphrases exact figures; golden answers require verbatim numbers like ₹18,00,000

---

## Project structure

```
DecideIQ/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routers (auth, sessions, documents, questions, comparison, evaluation)
│   │   ├── agents/           # LangGraph agents (question_generator)
│   │   ├── comparison/       # engine, answerer, scorer, verdict
│   │   ├── core/             # config (pydantic-settings), database (Motor)
│   │   ├── evaluation/       # ragas_eval.py
│   │   ├── ingestion/        # pdf_parser, html_cleaner, image_ocr, text_extractor
│   │   └── rag/              # pipeline, chunker, embedder, vector_store, retriever,
│   │                         # reranker, compressor, grounding, crag, hybrid_retriever
│   ├── tests/
│   │   └── test_ragas_golden.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.jsx              # Landing page
│   │   ├── login/page.jsx
│   │   ├── register/page.jsx
│   │   ├── dashboard/page.jsx
│   │   ├── sessions/
│   │   │   ├── new/page.jsx      # 5-step wizard
│   │   │   └── [id]/page.jsx     # Results dashboard
│   │   └── layout.jsx
│   ├── lib/api.js                # Typed API client
│   └── public/logo.svg
├── data/
│   └── job_offers/               # Sample documents for testing
├── docker-compose.yml
└── README.md
```

---

## Running locally

### Prerequisites
- Docker Desktop
- A `.env` file in the project root with:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MONGODB_URI=mongodb+srv://...
JWT_SECRET=your-secret-key
TAVILY_API_KEY=tvly-...
```

### Start

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Run RAGAS golden benchmark

```bash
docker exec decideiq-backend-1 python tests/test_ragas_golden.py
```

---

## Deployment

### Backend → Render
1. New Web Service → connect GitHub repo
2. Root directory: `backend` | Runtime: Docker
3. Add all `.env` keys as environment variables
4. Copy the service URL (e.g. `https://decideiq-api.onrender.com`)

### Frontend → Vercel
1. New Project → import GitHub repo
2. Root directory: `frontend` | Framework: Next.js
3. Add `NEXT_PUBLIC_API_URL=https://decideiq-api.onrender.com`
4. Deploy

---

## Key design decisions

- **Document-grounded question generation** - 5 questions are generated per document from its actual content, then deduplicated across all docs; this surfaces what each option actually covers rather than guessing from the title alone
- **CRAG with grounding check** - web search results are only used if they pass the same cosine similarity threshold (≥ 0.35) as RAG results; avoids hallucination from off-topic web results
- **Parent-child chunking** - children (400 chars) are retrieved for precision; parents (1800 chars) are returned for answering, giving full context without noise
- **All-not-found handling** - when every document returns "not found" for a question, scores default to neutral (5/10) so unanswerable questions don't penalise all options equally
- **FlashRank pre-downloaded at build time** - the 21.6 MB cross-encoder model is baked into the Docker image at `/models/flashrank`; no runtime download, no corruption from concurrent extraction

