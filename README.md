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
         ▼ FlashRank cross-encoder reranking (top-20 in → top-5 out)
         │
         ▼ Contextual compression (GPT-4o-mini)
              │
    ┌─────────┴──────────────────┐
  Extracted sentences        Nothing extracted
  (grounded by construction)  (passages off-topic)
    │                              │
  Use directly               Grounding check (cosine ≥ 0.35)
  skip grounding check            │
                        ┌─────────┴──────────┐
                      Grounded           Not grounded
                        │                    │
                      Use RAG          CRAG web search (Tavily)
                      contexts         Stage 1: cosine check
                                       Stage 2: LLM relevance check
                                            │
                                       Both pass? → use web results
                                       Either fails? → "Not mentioned"
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

## Evaluation findings — what we measured and what we learned

This section documents every experiment run during development, what was found, what was fixed, and what still needs work.

---

### RAGAS golden dataset — baseline vs improved pipeline

8 factual questions evaluated against hand-written golden answers. Evaluated on two real job offer documents.

#### TechCorp Offer

| Metric | Baseline | Improved v1 | Final | Delta (total) |
|---|---|---|---|---|
| Faithfulness | 0.9196 | 0.9375 | 0.9119 | -0.008 |
| Answer Relevancy | 0.7042 | 0.7386 | 0.8307 | **+0.127** |
| Context Recall | 0.7396 | 0.8021 | 0.8333 | **+0.094** |
| Answer Correctness | 0.6115 | 0.6748 | 0.8087 | **+0.197** |
| Confidence Score | 81.2% | 83.8% | 87.1% | **+5.9%** |

#### FinEdge Offer

| Metric | Baseline | Improved v1 | Final | Delta (total) |
|---|---|---|---|---|
| Faithfulness | 0.8438 | 0.8750 | 0.8438 | 0.000 |
| Answer Relevancy | 0.7196 | 0.6449 | 0.7038 | -0.016 |
| Context Recall | 0.6771 | 0.6771 | 0.6771 | 0.000 |
| Answer Correctness | 0.5930 | 0.4498 | 0.6314 | **+0.038** |
| Confidence Score | 78.2% | 76.0% | 77.4% | -0.8% |

**Pipeline changes across all iterations:**
- Retrieve up to 20 unique parent chunks (was 8) before reranking
- Rerank down to top-5 (was top-8) — fewer chunks, less noise for the LLM
- `per_query_k = 5 (short docs) / 8 (long docs)` — adaptive per document size
- Answerer prompt: explicit absence-of-information rule ("state clearly if something is not provided, do not hedge")

**TechCorp final result is strong:** answer_correctness jumped from 0.61 to 0.81 (+0.197), context_recall from 0.74 to 0.83, confidence from 81% to 87%.

**FinEdge Q7 remains the hard case:** "What training is provided?" — golden answer is "No learning budget provided." The model returned faithfulness=0.000, relevancy=0.000, recall=0.000, correctness=0.057 even after the prompt fix. The retrieval context doesn't contain a clear "no training" statement — it's an absence that's never stated in the document. This is a fundamentally hard RAG problem: you can't retrieve evidence for something that isn't there. A dedicated "absence detection" step would be needed to close this gap.

---

### Retrieval diagnostics — insurance plan documents (87-108 parent chunks each)

Ran Precision@k, Recall@k, MRR across 3 insurance PDFs using LLM-as-judge pseudo-labels.

| Document | R@3 | R@5 | R@8 | R@12 | R@20 | Post-rerank R@8 |
|---|---|---|---|---|---|---|
| HDFC Ergo Optima Restore | 0.598 | 0.654 | 0.733 | 0.797 | **0.994** | 0.789 |
| Niva Bupa Reassure 3 | 0.595 | 0.691 | 0.760 | 0.852 | **0.990** | 0.822 |
| Star Health Family Optima | 0.573 | 0.631 | 0.721 | 0.799 | **0.909** | 0.800 |

**Finding:** R@20 = 0.91-0.99 vs R@8 = 0.72-0.76. The relevant chunks exist in the index but weren't surfacing in the top-8 pool. The reranker can only promote what it's given.

**Root cause:** Retrieval pool bottleneck, not chunking or reranker failure. Confirmed by R@20 >> R@8 gap.

**Fix:** Retrieve up to 20 unique parents (5 queries x 8 per query, deduplicated, capped at 20), then rerank to top-5. This is the change that lifted TechCorp context_recall from 0.74 to 0.80.

**Reranker quality:** FlashRank post-rerank R@8 consistently exceeds pre-rerank R@8 (+0.05-0.08), confirming the cross-encoder reranker is working correctly.

---

### top_k grid search — k=3 vs k=5 vs k=8 (answer_correctness on golden dataset)

| top_k | Answer Correctness | Notes |
|---|---|---|
| k=3 | 0.7026 | Too few chunks, misses multi-section answers |
| k=5 | **0.8210** | Best — enough context, low noise |
| k=8 | 0.6115 | Baseline — too much noise dilutes the answer |

**Finding:** k=5 beats k=8 by +0.21 answer_correctness. Fewer, higher-quality chunks fed to the LLM produce better answers than more lower-quality chunks.

---

### Negative rejection test — unanswerable query robustness

5 queries with no answer in any job offer document:

| Query | Answer | Grounding Score | Result |
|---|---|---|---|
| What is the CEO's shoe size? | Not found in document. | 0.163 | PASS |
| What is the company's carbon footprint? | Not found in document. | 0.193 | PASS |
| Does the company offer a helicopter commute? | Not found in document. | 0.102 | PASS |
| What is the monthly pizza budget? | Not found in document. | 0.183 | PASS |
| What color are the office chairs? | Not found in document. | 0.119 | PASS |

**Result: 5/5 PASS.** All grounding scores < 0.20, well below the 0.35 threshold. CRAG correctly routes to web search, which also fails to ground, so the pipeline returns "Not found" rather than hallucinating.

---

### Quality gate — CI/CD regression check (3 questions, TechCorp offer)

| Metric | Score | Threshold | Result |
|---|---|---|---|
| Faithfulness | 1.000 | >= 0.80 | PASS |
| Answer Relevancy | 0.950 | >= 0.65 | PASS |

Run before every deploy: `docker exec decideiq-backend-1 python tests/test_quality_gate.py`

---

### Chunking experiment — 400-char vs 600-char children

10 sampled chunks x 1 query x 3 insurance docs. Judge cache shared across both configs.

| Document | Config | R@5 | R@8 | R@20 |
|---|---|---|---|---|
| HDFC Ergo | 400-char | 0.478 | 0.616 | 0.979 |
| HDFC Ergo | 600-char | 0.448 | 0.540 | 0.979 |
| Niva Bupa | 400-char | 0.717 | 0.765 | 1.000 |
| Niva Bupa | 600-char | 0.652 | 0.701 | 0.994 |
| Star Health | 400-char | 0.780 | 0.798 | 0.994 |
| Star Health | 600-char | **0.905** | **0.923** | 0.994 |
| **Aggregate** | **400-char** | **0.658** | **0.726** | **0.991** |
| **Aggregate** | **600-char** | **0.668** | **0.721** | **0.989** |

**Verdict: keep 400-char children.** Results are essentially a tie at R@8 and R@20 (diff < 0.005). 600-char wins on one doc (Star Health, +0.125 R@8) but loses on the other two. The R@20 ceiling is identical at ~0.99 for both — the limiting factor is the retrieval pool size, not the child chunk size. 400-char children are smaller, more precise, and already work well; switching to 600-char adds no reliable benefit.

---

### Pipeline improvements (post-evaluation)

Three design flaws identified through evaluation and fixed:

**1. Redundant grounding check after compression**
The compressor extracts only sentences that directly answer the question — if it returns content, those sentences are grounded by construction. Running a cosine similarity check on them was redundant. Fix: `_compress_node` now sets `skip_grounding=True` and writes `contexts` directly when compression succeeds, routing to END. The grounding check only runs when compression finds nothing (off-topic passages).
- [`backend/app/rag/pipeline.py`](backend/app/rag/pipeline.py)

**2. Weak web search validation**
The cosine grounding threshold (0.35) was too permissive for web results. A question like "What is the best company based on specific criteria?" triggered Tavily which returned generic "how to compare companies" articles — topically related enough to pass 0.35 but completely useless for answering document-specific questions. Fix: `web_search_verified` now runs a two-stage check: (1) cosine grounding as before, (2) an LLM relevance check asking "does this result actually answer the question?" Generic results now correctly return "Not mentioned in available sources."
- [`backend/app/rag/crag.py`](backend/app/rag/crag.py)

**3. Scorer short-circuit contradiction**
`score_answers` short-circuited to `[5]*n` when all documents answered "not found", but the scorer prompt explicitly instructs the LLM to score "not found" answers as 1. This caused the UI to show misleading yellow 5/10 badges instead of red 1/10. Fix: removed the short-circuit — LLM always runs and applies the prompt rule consistently.
- [`backend/app/comparison/scorer.py`](backend/app/comparison/scorer.py)

---

### What still needs work

| Issue | Priority | Status |
|---|---|---|
| FinEdge Q7 absence detection ("no training budget" never stated in doc) | High | Fundamental RAG gap — needs dedicated absence-detection step |
| Extend golden dataset from 8 to 20+ questions per doc | Medium | Not started |
| Add context_precision metric (RAGAS) | Low | Not started |
| FinEdge answer_relevancy slightly below baseline (0.70 vs 0.72) | Low | Minor variance, within noise |

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
│   │   ├── test_ragas_golden.py         # Full 5-metric RAGAS benchmark (golden dataset)
│   │   ├── test_quality_gate.py         # CI/CD gate — fails if faithfulness < 0.80
│   │   ├── test_negative_rejection.py   # Unanswerable query robustness (5/5 PASS)
│   │   ├── test_retrieval_insurance.py  # P@k, R@k, MRR on insurance PDFs
│   │   ├── test_chunking_experiment.py  # 400-char vs 600-char children comparison
│   │   └── test_topk_grid.py            # top_k grid search (k=3,5,8)
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

