from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from src.core.base import ProjectConfig, SourceDocument
from src.core.settings import AppSettings


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL = "Qdrant/bm25"


def build_index(
    documents: list[SourceDocument],
    config: ProjectConfig,
    settings: AppSettings,
) -> QdrantClient:
    client = build_qdrant_client(settings)

    try:
        client.delete_collection(collection_name=config.collection_name)
    except Exception:
        pass

    vectors_config = {
        DENSE_VECTOR_NAME: models.VectorParams(
            size=settings.embedding_output_dimensionality,
            distance=models.Distance.COSINE,
        )
    }

    sparse_vectors_config = None

    if settings.query_mode == "hybrid":
        sparse_vectors_config = {
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }

    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    chunks = _prepare_chunks(documents, settings)

    if not chunks:
        raise RuntimeError("No chunks were produced from the ingested documents.")

    points = []

    for text, metadata in chunks:
        vectors: dict[str, Any] = {
            DENSE_VECTOR_NAME: models.Document(
                text=text,
                model=settings.embedding_model,
            )
        }

        if settings.query_mode == "hybrid":
            vectors[SPARSE_VECTOR_NAME] = models.Document(
                text=text,
                model=SPARSE_MODEL,
            )

        payload = {
            "document": text,
            **metadata,
        }

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload=payload,
            )
        )

    client.upload_points(
        collection_name=config.collection_name,
        points=points,
        batch_size=settings.embedding_batch_size,
    )

    return client


def load_index(
    config: ProjectConfig,
    settings: AppSettings,
) -> QdrantClient:
    return build_qdrant_client(settings)


def collection_exists(
    config: ProjectConfig,
    settings: AppSettings,
) -> bool:
    client = build_qdrant_client(settings)

    try:
        client.get_collection(config.collection_name)
        return True
    except Exception:
        return False


def build_qdrant_client(settings: AppSettings) -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        local_inference_batch_size=settings.embedding_batch_size,
    )


def _prepare_chunks(
    documents: list[SourceDocument],
    settings: AppSettings,
) -> list[tuple[str, dict[str, Any]]]:
    chunks: list[tuple[str, dict[str, Any]]] = []

    docling_chunker = None

    for document in documents:
        if document.docling_document is not None:
            if docling_chunker is None:
                docling_chunker = _build_docling_chunker(settings)

            for chunk in docling_chunker.chunk(
                dl_doc=document.docling_document
            ):
                text = docling_chunker.contextualize(chunk=chunk).strip()

                if not text:
                    continue

                metadata = dict(document.metadata)
                chunk_metadata = chunk.meta.export_json_dict()

                headings = chunk_metadata.get("headings")
                if headings:
                    metadata["headings"] = headings

                page = _extract_page_number(chunk_metadata)
                if page is not None:
                    metadata["page"] = page

                chunks.append((text, metadata))

        else:
            for text in _chunk_plain_text(
                document.text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ):
                if text.strip():
                    chunks.append(
                        (
                            text.strip(),
                            dict(document.metadata),
                        )
                    )

    return chunks

def _build_docling_chunker(settings: AppSettings):
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(
    settings.embedding_model
    )

    model_max_tokens = hf_tokenizer.model_max_length
    tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=min(settings.chunk_size, model_max_tokens),

    )

    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )


def _chunk_plain_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    overlap = min(chunk_overlap, chunk_size - 1)
    step = max(1, chunk_size - overlap)

    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def _extract_page_number(chunk_metadata: dict[str, Any]) -> int | None:
    for doc_item in chunk_metadata.get("doc_items", []):
        for provenance in doc_item.get("prov", []):
            page_no = provenance.get("page_no")

            if page_no is not None:
                return page_no

    return None