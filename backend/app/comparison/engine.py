import asyncio
from app.rag.pipeline import run_pipeline
from app.comparison.answerer import answer_question
from app.comparison.scorer import score_answers, _is_not_found


async def run_comparison(session_id: str, session: dict) -> dict:
    docs = session["documents"]
    questions = session.get("generated_questions", [])

    if not questions:
        raise ValueError("No questions found. Run question generation first.")
    if len(docs) < 2:
        raise ValueError("At least 2 documents are required for comparison.")

    n_docs = len(docs)
    doc_raw_scores = [0] * n_docs
    answered_questions = 0   # questions where at least one doc had a real answer
    question_results = []

    for question in questions:
        # Retrieve context for every document concurrently
        pipeline_tasks = [
            run_pipeline(
                session_id=session_id,
                doc_idx=i,
                raw_text=doc["raw_text"],
                question=question,
                will_use_rag=doc.get("will_use_rag", False),
            )
            for i, doc in enumerate(docs)
        ]
        pipeline_results = await asyncio.gather(*pipeline_tasks)

        # Answer the question per document concurrently
        answer_tasks = [
            answer_question(question, pr["contexts"])
            for pr in pipeline_results
        ]
        answers = await asyncio.gather(*answer_tasks)

        # Score all answers comparatively in one LLM call
        scores = await score_answers(question, list(answers))

        all_not_found = all(_is_not_found(a) for a in answers)
        if not all_not_found:
            answered_questions += 1
        for i, score in enumerate(scores):
            doc_raw_scores[i] += score

        question_results.append(
            {
                "question": question,
                "per_doc": [
                    {
                        "doc_name": docs[i]["name"],
                        "answer": answers[i],
                        "score": scores[i],
                        "grounding_score": pipeline_results[i]["grounding_score"],
                        "source": pipeline_results[i]["source"],
                        # stored so RAGAS evaluation has the actual retrieved passages
                        "contexts": pipeline_results[i]["contexts"],
                    }
                    for i in range(n_docs)
                ],
            }
        )

    # Use only answerable questions in the denominator; unanswerable ones contribute
    # equal 5s to all docs so they don't change relative rankings but do inflate max_possible.
    effective_questions = max(answered_questions, 1)
    max_possible = effective_questions * 10 + (len(questions) - effective_questions) * 5
    doc_summaries = [
        {
            "doc_name": docs[i]["name"],
            "raw_score": doc_raw_scores[i],
            "percentage": round(doc_raw_scores[i] / max_possible * 100, 1),
        }
        for i in range(n_docs)
    ]

    winner_idx = max(range(n_docs), key=lambda i: doc_raw_scores[i])

    return {
        "question_results": question_results,
        "doc_summaries": doc_summaries,
        "winner_idx": winner_idx,
        "winner_name": docs[winner_idx]["name"],
    }
