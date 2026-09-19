from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient

from src.core.base import ProjectConfig
from src.core.retriever import RetrievedChunk, retrieve_chunks
from src.core.schemas import QueryArtifacts, RAGResponse
from src.core.settings import AppSettings


def answer_question(
    index: QdrantClient,
    question: str,
    config: ProjectConfig,
    settings: AppSettings,
) -> QueryArtifacts:
    retrieved_chunks = retrieve_chunks(
        index=index,
        question=question,
        config=config,
        settings=settings,
    )

    answer = _generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        config=config,
        settings=settings,
    )

    sources = _dedupe_sources(retrieved_chunks)
    retrieval_context = _context_snippets(retrieved_chunks)
    evidence = _build_evidence_summary(retrieved_chunks)
    confidence = _infer_confidence(retrieved_chunks)

    return QueryArtifacts(
        response=RAGResponse(
            answer=answer,
            evidence=evidence,
            sources=sources,
            confidence=confidence,
            disclaimer=config.disclaimer,
        ),
        retrieval_context=retrieval_context,
    )


def _generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    config: ProjectConfig,
    settings: AppSettings,
) -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to the .env file before generating answers."
        )
    client = OpenAI(
        api_key=groq_api_key,
        base_url=settings.groq_base_url,
    )

    context = _build_llm_context(retrieved_chunks)

    user_prompt = (
        "Retrieved context:\n\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}"
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": config.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        return "No answer could be generated from the retrieved evidence."

    return content.strip()


def _build_llm_context(
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    if not retrieved_chunks:
        return "No supporting context was retrieved."

    sections = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.metadata

        source_org = (
            metadata.get("source_org")
            or metadata.get("source")
            or "Unknown source"
        )

        source_file = (
            metadata.get("source_file")
            or metadata.get("title")
            or "Unknown document"
        )

        page = metadata.get("page")

        source_label = f"{source_org}: {source_file}"

        if page is not None:
            source_label = f"{source_label} (page {page})"

        sections.append(
            f"[Evidence {index}]\n"
            f"Source: {source_label}\n"
            f"{chunk.text}"
        )

    return "\n\n".join(sections)


def _dedupe_sources(
    retrieved_chunks: list[RetrievedChunk],
) -> list[str]:
    seen: list[str] = []

    for chunk in retrieved_chunks:
        metadata = chunk.metadata

        source_org = (
            metadata.get("source_org")
            or metadata.get("source")
            or "Unknown source"
        )

        source_file = (
            metadata.get("source_file")
            or metadata.get("title")
            or "Unknown document"
        )

        page = metadata.get("page")

        label = f"{source_org}: {source_file}"

        if page is not None:
            label = f"{label} (page {page})"

        if label not in seen:
            seen.append(label)

    return seen


def _context_snippets(
    retrieved_chunks: list[RetrievedChunk],
) -> list[str]:
    snippets: list[str] = []

    for chunk in retrieved_chunks[:4]:
        snippet = " ".join(chunk.text.split())

        if snippet:
            snippets.append(snippet[:500])

    return snippets


def _build_evidence_summary(
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    if not retrieved_chunks:
        return "No supporting evidence was retrieved."

    snippets = _context_snippets(retrieved_chunks[:2])

    if snippets:
        return " | ".join(snippets)

    return "Supporting evidence was retrieved."


def _infer_confidence(
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    count = len(retrieved_chunks)

    if count >= 4:
        return "high"

    if count >= 2:
        return "moderate"

    return "low"