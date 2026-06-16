from __future__ import annotations
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from app.core.config import settings

# Production: faithfulness + answer_relevancy — both work without golden answers.
# LLMContextPrecisionWithoutReference exists in ragas 0.2.x but its required_columns
# property conflicts with validate_required_columns in 0.2.14 — excluded until upgrade.
_PRODUCTION_METRICS = [faithfulness, answer_relevancy]

# Full eval adds context_recall + answer_correctness — both REQUIRE a reference (golden answer).
_FULL_METRICS = [faithfulness, answer_relevancy, context_recall, answer_correctness]


def _configure_metrics(full: bool = False) -> None:
    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)
    )
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    )
    metrics = _FULL_METRICS if full else _PRODUCTION_METRICS
    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = emb


def build_ragas_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    references: list[str] | None = None,
) -> Dataset:
    data: dict = {"question": questions, "answer": answers, "contexts": contexts}
    if references:
        data["reference"] = references
    return Dataset.from_dict(data)


def _extract_scores(scores_df, questions: list[str], metric_cols: list[str]) -> dict:
    result: dict = {}
    for col in metric_cols:
        if col in scores_df.columns:
            result[col] = round(float(scores_df[col].mean()), 4)

    # Confidence score uses the two reference-free metrics only
    faith = result.get("faithfulness", 0.0)
    rel = result.get("answer_relevancy", 0.0)
    result["confidence_score"] = round((faith + rel) / 2, 4)

    result["per_question"] = [
        {
            "question": questions[i],
            **{
                col: round(float(scores_df.iloc[i][col]), 4)
                for col in metric_cols
                if col in scores_df.columns
            },
        }
        for i in range(len(questions))
    ]
    return result


def run_ragas(dataset: Dataset) -> dict:
    """
    Production eval — no golden answers needed.
    Metrics: faithfulness, answer_relevancy, context_precision (LLM-as-judge).
    Confidence score = (faithfulness + answer_relevancy) / 2.
    """
    _configure_metrics(full=False)
    result = evaluate(
        dataset=dataset,
        metrics=_PRODUCTION_METRICS,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=2, max_retries=3, timeout=60),
    )
    scores = result.to_pandas()
    questions = dataset["question"]
    metric_cols = ["faithfulness", "answer_relevancy"]
    return _extract_scores(scores, questions, metric_cols)


def run_ragas_full(dataset: Dataset) -> dict:
    """
    Full eval — requires 'reference' (golden answer) column in dataset.
    Metrics: faithfulness, answer_relevancy, context_precision,
             context_recall, answer_correctness.
    Use this for development benchmarking and resume-worthy RAGAS reports.
    """
    if "reference" not in dataset.column_names:
        raise ValueError("Full eval requires a 'reference' column with golden answers.")
    _configure_metrics(full=True)
    result = evaluate(
        dataset=dataset,
        metrics=_FULL_METRICS,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=2, max_retries=3, timeout=60),
    )
    scores = result.to_pandas()
    questions = dataset["question"]
    metric_cols = [
        "faithfulness", "answer_relevancy",
        "context_recall", "answer_correctness",
    ]
    return _extract_scores(scores, questions, metric_cols)
