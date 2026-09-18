from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from src.core.base import ProjectConfig
from src.core.indexer import (
    DENSE_VECTOR_NAME,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
)
from src.core.settings import AppSettings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float | None = None


def retrieve_chunks(
    index: QdrantClient,
    question: str,
    config: ProjectConfig,
    settings: AppSettings,
) -> list[RetrievedChunk]:
    if settings.query_mode == "hybrid":
        return _retrieve_hybrid(
            client=index,
            question=question,
            config=config,
            settings=settings,
        )

    return _retrieve_dense(
        client=index,
        question=question,
        config=config,
        settings=settings,
    )


def _retrieve_dense(
    client: QdrantClient,
    question: str,
    config: ProjectConfig,
    settings: AppSettings,
) -> list[RetrievedChunk]:
    response = client.query_points(
        collection_name=config.collection_name,
        query=models.Document(
            text=question,
            model=settings.embedding_model,
        ),
        using=DENSE_VECTOR_NAME,
        limit=settings.similarity_top_k,
        with_payload=True,
        with_vectors=False,
    )

    return [_to_retrieved_chunk(point) for point in response.points]


def _retrieve_hybrid(
    client: QdrantClient,
    question: str,
    config: ProjectConfig,
    settings: AppSettings,
) -> list[RetrievedChunk]:
    dense_response = client.query_points(
        collection_name=config.collection_name,
        query=models.Document(
            text=question,
            model=settings.embedding_model,
        ),
        using=DENSE_VECTOR_NAME,
        limit=settings.similarity_top_k,
        with_payload=True,
        with_vectors=False,
    )

    sparse_response = client.query_points(
        collection_name=config.collection_name,
        query=models.Document(
            text=question,
            model=SPARSE_MODEL,
        ),
        using=SPARSE_VECTOR_NAME,
        limit=settings.sparse_top_k,
        with_payload=True,
        with_vectors=False,
    )

    fused_points = _weighted_reciprocal_rank_fusion(
        dense_points=dense_response.points,
        sparse_points=sparse_response.points,
        dense_weight=settings.hybrid_alpha,
        limit=settings.similarity_top_k,
    )

    return [
        _to_retrieved_chunk(point, score=fused_score)
        for point, fused_score in fused_points
    ]


def _weighted_reciprocal_rank_fusion(
    dense_points,
    sparse_points,
    dense_weight: float,
    limit: int,
):
    dense_weight = max(0.0, min(1.0, dense_weight))
    sparse_weight = 1.0 - dense_weight

    rank_constant = 60
    fused_scores: dict[str, float] = {}
    points_by_id = {}

    for rank, point in enumerate(dense_points, start=1):
        point_id = str(point.id)
        points_by_id[point_id] = point

        fused_scores[point_id] = fused_scores.get(point_id, 0.0) + (
            dense_weight / (rank_constant + rank)
        )

    for rank, point in enumerate(sparse_points, start=1):
        point_id = str(point.id)
        points_by_id[point_id] = point

        fused_scores[point_id] = fused_scores.get(point_id, 0.0) + (
            sparse_weight / (rank_constant + rank)
        )

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )[:limit]

    return [
        (points_by_id[point_id], fused_scores[point_id])
        for point_id in ranked_ids
    ]


def _to_retrieved_chunk(
    point,
    score: float | None = None,
) -> RetrievedChunk:
    payload = dict(point.payload or {})
    text = str(payload.pop("document", ""))

    resolved_score = score
    if resolved_score is None and point.score is not None:
        resolved_score = float(point.score)

    return RetrievedChunk(
        text=text,
        metadata=payload,
        score=resolved_score,
    )