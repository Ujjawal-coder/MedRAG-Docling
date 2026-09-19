from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from eval.shared.groq_judge import GroqJudgeModel


def _named(metric: object, name: str) -> object:
    setattr(metric, "name", name)
    return metric


def build_rag_metrics():
    judge_model = GroqJudgeModel()

    return [
        _named(
            AnswerRelevancyMetric(
                threshold=0.7,
                model=judge_model,
            ),
            "Answer Relevance",
        ),
        _named(
            ContextualRelevancyMetric(
                threshold=0.7,
                model=judge_model,
            ),
            "Context Relevance",
        ),
        _named(
            FaithfulnessMetric(
                threshold=0.7,
                model=judge_model,
            ),
            "Groundedness",
        ),
    ]