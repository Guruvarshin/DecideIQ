# DecideIQ — Interview Preparation

Every question you could realistically be asked about this project, grouped by topic.

---

## 1. Project Overview

**Q: Walk me through DecideIQ in one minute.**
A: DecideIQ is a full-stack AI application that compares multiple documents side-by-side — job offers, insurance plans, contracts — and gives you a scored verdict with evidence. You upload 2+ documents, optionally add your own questions, and the system generates document-grounded questions, runs each question through a RAG pipeline per document, scores every answer 1-10 comparatively, and has Claude Sonnet write a final verdict naming the winner. It's deployed on Render (backend) and Vercel (frontend).

**Q: Why did you build this? What problem does it solve?**
A: Most people comparing job offers or insurance plans read them linearly and lose track of specifics. DecideIQ forces the comparison to be structured — same questions, same scoring rubric, evidence sourced from the documents themselves rather than memory.

**Q: What were the hardest engineering problems you solved?**
A: Three main ones — (1) the retrieval pool bottleneck where R@8=0.73 but R@20=0.99 on insurance documents, meaning relevant chunks existed but never surfaced; (2) scoring bugs where all documents got 5/10 when questions were unanswerable; (3) corrupted multi-byte characters in JSX files caused by encoding issues that required a full file rewrite instead of a patch.

---

## 2. RAG Pipeline Architecture

**Q: Explain your RAG pipeline end to end.**
A: A user question enters a LangGraph stateful graph. First, multi-query decomposition generates 3 sub-queries, and HyDE creates a hypothetical document embedding — both run in parallel. Then hybrid retrieval runs BM25 and dense (ChromaDB cosine) search for each query, fuses results with RRF (k=60), expands child chunks to parent chunks, and caps at 20 unique parents. FlashRank cross-encoder reranks those 20 down to top-5. Contextual compression (GPT-4o-mini) drops irrelevant sentences. A grounding check (cosine similarity ≥ 0.35) decides whether to use RAG context or fall back to Tavily web search. The web result goes through the same grounding check — if it also fails, the pipeline returns "Not mentioned in available sources."

**Q: Why LangGraph instead of a simple function chain?**
A: LangGraph gives you typed state that flows through every node without passing arguments explicitly, conditional routing (RAG vs full-context vs web fallback) without nested if-else, and easy addition of new nodes without refactoring existing ones. The state TypedDict also makes it obvious what data each node needs and produces.

**Q: What is HyDE and why did you use it?**
A: Hypothetical Document Embedding — instead of embedding the raw question (which is short and syntactically different from document text), you ask the LLM to generate a hypothetical answer paragraph, then embed that. The resulting vector is closer to how the relevant document text is phrased, improving dense retrieval recall. It runs in parallel with multi-query so there's no latency cost.

**Q: What is multi-query decomposition?**
A: A single question like "what is the total compensation?" can miss chunks that only mention "bonus" or "ESOP" separately. Multi-query generates 3 sub-queries that rephrase or decompose the original — each sub-query retrieves independently, results are deduplicated and fused. This improves recall without increasing final context size.

**Q: What is RRF and why k=60?**
A: Reciprocal Rank Fusion combines ranked lists from BM25 and dense retrieval by scoring each item as the sum of 1/(k + rank) across all lists. k=60 is a well-established default from the original RRF paper — it smooths out rank differences at the top without over-penalizing items that appear in only one list. The formula means rank-1 in one list gets 1/61 ≈ 0.016, and even rank-20 gets 1/80 ≈ 0.012, so cross-list agreement matters more than individual rank dominance.

**Q: Explain parent-child chunking. Why not just chunk once?**
A: Children (400 chars) are small enough that retrieval hits the precise passage the question is about. But answering a question from 400 chars often lacks context — the salary might be in one child, the bonus in the next. Parent chunks (1800 chars) span a full section. Retrieval uses children for precision, but the LLM answerer sees parents for completeness. Without this, either retrieval is imprecise (large chunks) or answers are incomplete (small chunks).

**Q: Why 1800 chars for parent and 400 for child specifically?**
A: These are heuristic values calibrated for the kind of documents DecideIQ targets — job offers, insurance plans, contracts. A 1800-char parent is roughly one full section or clause. A 400-char child is roughly one or two sentences — precise enough for a cross-encoder to judge relevance. The chunking experiment comparing 400 vs 600-char children showed marginal R@5 gain (+0.022) but no meaningful improvement at R@8 or R@20, suggesting 400-char is already near-optimal for this domain.

**Q: What is the grounding check? How does it work?**
A: After compression, the pipeline embeds both the question and the compressed context, then computes cosine similarity between them. If the similarity is ≥ 0.35, the context is considered grounded (relevant enough to answer from). Below 0.35, the pipeline falls back to Tavily web search. The threshold was chosen empirically — 0.35 is low enough to accept valid but tersely phrased answers, high enough to reject off-topic retrievals.

**Q: What is CRAG?**
A: Corrective RAG — a pattern where the pipeline checks whether retrieved context is actually relevant before answering from it, and falls back to an external source (web search) if not. Standard RAG blindly uses whatever it retrieves. CRAG adds the grounding check as a quality gate.

**Q: Why FlashRank for reranking instead of a heavier model?**
A: FlashRank uses ms-marco-MiniLM-L-12-v2, a cross-encoder that runs fully locally with no API cost. It's fast enough for a web app (sub-second for 20 candidates), doesn't add latency like an API call would, and cross-encoders consistently outperform bi-encoders for reranking because they see the query and document together rather than separately embedded.

**Q: What is the difference between bi-encoder and cross-encoder retrieval?**
A: Bi-encoders embed query and document independently into vectors, then compare with cosine similarity — fast but less accurate because the two are never jointly contextualized. Cross-encoders concatenate query + document and classify relevance jointly — slower (can't precompute document vectors) but much more accurate. In this pipeline: dense retrieval uses bi-encoder (fast, pre-indexed), reranking uses cross-encoder (accurate, only on the short candidate list).

**Q: What is contextual compression and when does it help?**
A: After reranking, parent chunks can still contain irrelevant sentences. Contextual compression runs each chunk through GPT-4o-mini to keep only the sentences that directly address the question. This reduces noise in the context window, especially on dense documents where a single parent chunk covers multiple topics.

---

## 3. Chunking and Ingestion

**Q: How do you handle PDFs, HTML, images, and plain text differently?**
A: PDFs use PyMuPDF for structured text extraction. HTML uses BeautifulSoup to strip tags and clean the DOM. Images use Tesseract OCR via pytesseract. Plain text is read directly. All four paths produce a single `raw_text` string that feeds the same chunking pipeline downstream.

**Q: What happens if OCR produces noisy text?**
A: Tesseract output for low-quality scans can be noisy. The pipeline doesn't have a dedicated denoising step today — this is a known limitation. For well-scanned documents it works fine. A future improvement would be to run a cleanup pass with an LLM for low-confidence OCR output.

**Q: How does the vector store work?**
A: ChromaDB stores one collection per document (keyed by session_id + doc_idx). Each child chunk is stored with its text as the document and its embedding as the vector. At query time, ChromaDB returns the closest child chunks by cosine similarity. Parent text is attached to each Chunk object in memory and looked up by parent_index — it's not stored in ChromaDB separately.

---

## 4. Evaluation and Metrics

**Q: What is RAGAS and what does each metric measure?**
A: RAGAS is a framework for evaluating RAG systems without requiring human-labeled answers for every question.
- **Faithfulness**: are all claims in the answer grounded in the retrieved context? (hallucination check)
- **Answer Relevancy**: is the answer actually responsive to the question? (focuses on answer quality)
- **Context Recall**: what fraction of the golden answer's facts are present in the retrieved context? (retrieval completeness — requires golden answers)
- **Answer Correctness**: how factually aligned is the answer with the golden answer? (requires golden answers)

**Q: What is the difference between faithfulness and answer correctness?**
A: Faithfulness checks that the answer doesn't say things not in the retrieved context — it catches hallucination. Answer correctness checks that the answer matches the known correct answer — it catches omissions and paraphrasing errors. You can have high faithfulness but low correctness (the retrieved context was incomplete so the answer was grounded but wrong) or low faithfulness but high correctness (the LLM happened to say the right thing but made it up).

**Q: What is context recall and how is it measured in RAGAS?**
A: RAGAS breaks the golden answer into atomic facts, then checks what fraction of those facts are present somewhere in the retrieved context. A context recall of 0.80 means 80% of the facts in the ground truth answer were retrievable — the remaining 20% weren't in the top-5 chunks fed to the LLM.

**Q: What is Precision@k, Recall@k, MRR? How did you compute them?**
A: For a given query and a ranked list of retrieved chunks:
- **Precision@k**: fraction of top-k chunks that are relevant — measures how much noise is in the retrieved set
- **Recall@k**: fraction of all relevant chunks that appear in top-k — measures completeness of retrieval
- **MRR (Mean Reciprocal Rank)**: 1/rank of the first relevant result — measures how quickly the system surfaces a relevant chunk

You need relevance labels to compute these. Without ground truth, the project uses LLM-as-judge: GPT-4o-mini is asked "does this chunk help answer this query? yes/no" for each (query, chunk) pair. The source chunk (the one the query was generated from) is always marked relevant as a guaranteed positive.

**Q: What is LLM-as-judge and what are its limitations?**
A: LLM-as-judge uses an LLM to generate pseudo-labels for relevance instead of human annotators. It's much cheaper and faster than human labeling. Limitations: the LLM can be inconsistent across runs, it can miss subtle relevance, and it's biased toward longer more detailed chunks. Results should be treated as approximate, not ground truth.

**Q: What was the most important finding from your retrieval diagnostics?**
A: The retrieval pool bottleneck. On insurance documents with 87-108 parent chunks, R@8 was 0.72-0.76 but R@20 was 0.91-0.99. This means the relevant chunks existed in the index but weren't surfacing in the top-8. The reranker had no chance to promote them because they weren't in its input. Expanding the pool to 20 candidates before reranking fixed this — context recall improved from 0.74 to 0.80 on the golden benchmark.

**Q: Why did the top_k grid search find k=5 beats k=8?**
A: The reranker selects the best k chunks from the pool of 20. With k=8, the LLM answerer receives 8 chunks — some of which are lower relevance and introduce noise. The LLM gets confused or diluted. With k=5, only the most relevant chunks are passed — the answer is more focused and precise. Answer correctness improved by +0.21 (0.8210 vs 0.6115).

**Q: What does the quality gate test do exactly?**
A: It runs 3 fixed questions through the full RAG pipeline against TechCorp offer, evaluates with RAGAS (faithfulness + answer relevancy), and exits with code 1 if either score drops below threshold (faithfulness < 0.80, answer relevancy < 0.65). This means if a code change — a chunking change, a prompt edit, a pipeline modification — breaks the pipeline, it fails CI before deployment. Current scores: faithfulness 1.000, answer relevancy 0.950.

**Q: What is the negative rejection test?**
A: Five queries with no possible answer in the document ("CEO's shoe size", "office chair color", etc.) are sent through the full pipeline. The test asserts every answer contains "Not found in document." and grounding scores stay below 0.20. It passed 5/5 — the CRAG grounding check correctly routes these to "not found" rather than hallucinating. This tests the pipeline's robustness to garbage-in inputs.

**Q: Your FinEdge answer_correctness dropped from 0.59 to 0.45 after the pipeline improvement. How do you explain that?**
A: Context recall for FinEdge was unchanged at 0.677 — retrieval didn't regress. The drop came from two specific questions: "Is remote work permitted?" (golden: "No, not permitted — full office attendance required") and "What training is provided?" (golden: "No learning budget or internet allowance"). With the expanded context pool, the LLM received more context and started hedging ("the document does not explicitly mention...") rather than stating the absence directly. The fix is a prompt-level rule explicitly instructing the answerer to state absences clearly rather than hedge. This has been applied but not yet re-evaluated due to API quota.

---

## 5. Scoring and Verdict

**Q: How does the scoring work?**
A: After every question is answered for every document, all answers are sent to GPT-4o-mini together so it can score comparatively. The scorer sees all documents' answers to the same question at once and assigns 1-10 to each. Final percentage = sum of scores / (answered_questions × 10). Questions where every document returned "not found" are excluded from both numerator and denominator — they don't penalise anyone.

**Q: Why does the scorer see all answers at once rather than scoring each independently?**
A: Comparative scoring requires seeing all options together. If TechCorp offers ₹18L and FinEdge offers ₹20L, scoring them independently might give both 7/10. Seeing them together lets the scorer give TechCorp 6 and FinEdge 8 — making the difference visible. This is the whole point of a comparison tool.

**Q: How do you handle questions where no document has an answer?**
A: Before the fix, unanswerable questions used `max(answered_questions, 1)` as the denominator which gave wrong scores (effectively 5/10 for all). The fix: if all documents returned "not found" for a question, that question is completely excluded — it doesn't add to the numerator or the denominator. This makes the final percentage reflect only questions the documents actually addressed.

**Q: Why does the verdict use Claude Sonnet instead of GPT-4o-mini?**
A: The verdict is a qualitative written summary — it needs to weigh tradeoffs, acknowledge nuance, and write clearly. Claude Sonnet produces more coherent, balanced long-form prose for this kind of synthesis task. GPT-4o-mini is used for all structured, repeatable tasks (scoring, sub-query generation) where cost and speed matter more than prose quality.

---

## 6. Question Generation

**Q: How does question generation work?**
A: A 3-node LangGraph pipeline. Node 1 rephrases user-provided questions to be specific and measurable. Node 2 runs concurrently across all documents — each document gets 5 questions generated from its first 3000 chars of content (covering key decision dimensions: salary, benefits, leave, work model, etc.). Node 3 deduplicates across all per-document questions with an LLM, keeping the clearest phrasing of each unique question. Final output: rephrased user questions + deduplicated document-grounded questions.

**Q: Why generate questions from documents rather than just from the title?**
A: Title-based generation produces generic questions. "Compare these job offers" generates "What is the salary?" — useful but shallow. Document-based generation surfaces what each option actually contains — if one offer has ESOPs and the other doesn't, the system generates an ESOP question automatically. This makes the comparison relevant to the specific documents uploaded.

**Q: What is the risk of generating questions from document content?**
A: If a document mentions something unusual (e.g., a clawback clause), that question gets generated even if the user doesn't care about it. Deduplication also isn't perfect — similar questions can slip through. A better approach would be to weight question generation toward categories the user indicated matter most. Currently there's no user preference weighting.

---

## 7. System Design and Infrastructure

**Q: Why Docker Compose for local development?**
A: The backend has system dependencies (Tesseract, ChromaDB persistence, FlashRank models) that are annoying to set up locally. Docker bakes all of that in. docker compose up --build gives you a reproducible environment. The Dockerfile also bakes the FlashRank model at build time so there's no runtime download.

**Q: How is authentication implemented?**
A: JWT tokens (PyJWT HS256) stored in httpOnly cookies. httpOnly prevents JavaScript access, mitigating XSS token theft. SameSite=None + Secure is set for production (cross-origin between Vercel frontend and Render backend). The frontend uses `credentials: include` on all fetch calls so cookies are sent cross-origin. Passwords are hashed with bcrypt.

**Q: How does session and document storage work?**
A: Sessions and user metadata are stored in MongoDB Atlas (via Motor async client). Document raw text and extracted content are stored in MongoDB as part of the session document. Embeddings and chunk vectors are stored in ChromaDB (one collection per document). This separation means if you need to re-embed (e.g., model change), you only rebuild ChromaDB without touching MongoDB.

**Q: What happens if the Render backend goes cold (free tier sleep)?**
A: Render free tier spins down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds for the container to wake up, load the FlashRank model, and reconnect to MongoDB. Users see a slow first response. The fix is to keep the backend warm with a ping (not implemented — the free tier cost tradeoff is acceptable for a demo project).

**Q: How do you handle rate limits from OpenAI?**
A: RAGAS evaluation uses `RunConfig(max_workers=1, max_retries=5, timeout=120)` to serialize LLM calls. The insurance diagnostic uses a batched judge pattern — 20 LLM calls per batch with a 3-second pause between batches to stay under the 500 RPM limit. The daily RPD limit (10,000 requests) is a harder constraint — the project doesn't have a solution beyond using off-peak hours or upgrading the OpenAI tier.

---

## 8. Frontend

**Q: Why Next.js App Router?**
A: App Router supports server components (for auth checks and initial data fetching without client-side waterfalls), file-based routing, and layout nesting. For this project the main benefit is that the session results page can fetch comparison data server-side rather than loading a blank page and then fetching.

**Q: How does file upload work on the frontend?**
A: A multi-step wizard. Step 1: title + user questions. Step 2-4: file drop per document (PDF, HTML, TXT, image accepted). Step 5: generate questions and confirm. Files are uploaded to the FastAPI backend via `multipart/form-data`. The backend parses, chunks, embeds, and stores everything before responding. Progress is shown client-side during the "Uploading and indexing..." phase.

---

## 9. Failure Modes and Edge Cases

**Q: What happens if a document is in a language other than English?**
A: The pipeline would likely degrade. BM25 is language-agnostic but the dense embeddings (text-embedding-3-small) and all LLM prompts are English-optimized. RAGAS evaluation would also be unreliable. Multi-language support would require language detection and model selection at ingestion time.

**Q: What if the same question is unanswerable in all documents?**
A: The scoring engine detects this (`all_not_found = all(_is_not_found(a) for a in answers)`) and skips that question entirely — it contributes 0 to both numerator and denominator. The verdict prompt also receives the unanswered questions list so Claude Sonnet can note them.

**Q: What if a user uploads the same document twice?**
A: The pipeline would compare the document against itself. Scores would be identical, and the verdict would be meaningless. There's no deduplication check at upload time — a future improvement would be to hash document content and warn on duplicates.

**Q: What is the maximum document size the system handles?**
A: No hard limit is enforced, but very large documents (>100 pages) cause two problems: embedding all child chunks takes time and API tokens at ingestion, and the full_context path (used for short docs) would overflow the LLM context window. For long documents the pipeline always uses the RAG path (will_use_rag=True), which limits context to the top-5 reranked chunks regardless of document length.

**Q: What does will_use_rag control?**
A: For very short documents (under a certain word count), `will_use_rag=False` and the full document text is passed directly as context — no retrieval needed. For longer documents, `will_use_rag=True` triggers the full hybrid retrieval pipeline. This avoids retrieval overhead on 2-page documents where you could just pass everything.

---

## 10. What You Would Do Differently / Improvements

**Q: What would you improve if you had more time?**
A: Several things. (1) Re-evaluate FinEdge after the absence-of-info prompt fix to confirm correctness recovers. (2) Run the chunking experiment (400 vs 600-char) — early signal favors 400-char but only one doc's results came through. (3) Add context_precision to RAGAS evaluation — currently measuring recall but not precision of retrieved context. (4) Extend the golden dataset from 8 to 20+ questions per document for more statistically robust scores. (5) Add streaming responses on the frontend so users see answers appearing rather than waiting for all 8 questions to complete.

**Q: Would you use a different vector store in production?**
A: ChromaDB is excellent for a single-instance deployment but doesn't support horizontal scaling. For production with multiple backend instances, you'd switch to Pinecone, Weaviate, or Qdrant — all support multi-tenant namespaces (equivalent to per-doc collections) and distributed deployment. The rest of the pipeline would be unchanged.

**Q: Would you change the LLM used for scoring or answering?**
A: For answering, GPT-4o-mini works well for factual extraction. For scoring (comparative judgment), a stronger model like GPT-4o would likely reduce inconsistency on close comparisons. The verdict already uses Claude Sonnet. The tradeoff is cost — for a demo the current setup is fine, but for a paid product the scoring step should use a stronger model.

**Q: How would you scale this to handle 1000 concurrent users?**
A: The main bottlenecks are (1) OpenAI API rate limits — solution is per-user rate limiting and request queuing; (2) ChromaDB — replace with a distributed vector DB; (3) the RAG pipeline is fully async (all LangGraph nodes are async) so the backend scales horizontally with multiple Uvicorn workers already; (4) MongoDB Atlas scales automatically. The overall architecture is stateless at the FastAPI layer so Kubernetes horizontal pod autoscaling would work directly.

---

## 11. Behavioral / Learning Questions

**Q: What did you learn about RAG systems from building this?**
A: The biggest insight was that retrieval quality is the ceiling for answer quality — no matter how good the LLM is, it can only answer from what's retrieved. The specific lesson: pool size before reranking matters more than reranker quality. R@20 = 0.99 but R@8 = 0.73 — the relevant chunks were there, they just weren't in the pool the reranker saw. The second insight: fewer, higher-quality chunks (k=5) beats more lower-quality chunks (k=8) by +0.21 answer correctness — LLMs get confused by noise, not just by absence.

**Q: What surprised you most during evaluation?**
A: That FinEdge correctness went down after the pipeline improvement that helped TechCorp. The retrieval didn't change (context recall was flat at 0.677) — the regression was entirely in the LLM answerer hedging on absence-of-information answers. It showed that RAGAS metrics can move in different directions for the same pipeline change depending on what kind of questions are in the dataset. A small golden dataset (8 questions) amplifies individual question behavior.

**Q: How did you debug the scoring bug where all documents got 5/10?**
A: Added logging to print raw scores and effective_questions before the percentage calculation. Found that `effective_questions = max(answered_questions, 1)` was the culprit — when all answers were "not found", answered_questions was 0, max() set it to 1, and the division gave 0/10 = 0... but wait, actually scores were 5/10, which meant scores were 5 and the denominator was 10. Traced further and found the comparative scorer was returning 5 for each doc when all answers were "not found" (a reasonable LLM default), and the denominator was wrong. Fixed by excluding unanswerable questions from both numerator and denominator.

**Q: What was the hardest debugging session?**
A: The corrupted JSX symbols. Characters like `→`, `✓`, `×` appeared as `â†'`, `âœ"`, `Ã—` in the browser. Multiple attempts to fix this with Python re-encoding scripts (UTF-8 decode → latin-1 encode sequences) kept failing or producing different corruptions. Eventually diagnosed it as the file being saved with UTF-8 BOM or double-encoded at some point in the edit history. The only reliable fix was a complete file rewrite with pure ASCII replacements (`>` instead of `→`, `x` instead of `×`) — no multi-byte characters at all.
