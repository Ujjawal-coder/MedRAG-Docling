from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    docling_document: Any | None = None


@dataclass(frozen=True)
class MetadataField:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    collection_name: str
    system_prompt: str
    disclaimer: str
    data_dir: Path
    metadata_fields: list[MetadataField]
    golden_dataset_path: Path


class DocumentIngestor(ABC):
    """Project-specific parsing contract."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    @abstractmethod
    def load_and_parse(self) -> list[SourceDocument]:
        """Load raw data, parse it, and return source documents."""

    @abstractmethod
    def enrich_metadata(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        """Attach domain-specific metadata needed by the shared pipeline."""

    def ingest(self) -> list[SourceDocument]:
        return self.enrich_metadata(self.load_and_parse())