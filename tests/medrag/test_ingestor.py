from pathlib import Path
from types import SimpleNamespace

from src.projects.medrag.ingestor import MedRAGIngestor


class DummyDocument:
    def __init__(self, metadata):
        self.metadata = metadata


def test_enrich_metadata_maps_pubmed_documents():
    ingestor = MedRAGIngestor(config=None)  # type: ignore[arg-type]
    doc = DummyDocument({"source": "pubmed", "query": "hypertension management guideline"})

    enriched = ingestor.enrich_metadata([doc])[0]

    assert enriched.metadata["source_org"] == "PubMed"
    assert enriched.metadata["evidence_type"] == "research_abstract"
    assert enriched.metadata["specialty"] == "cardiology"


def test_enrich_metadata_maps_fda_labels():
    ingestor = MedRAGIngestor(config=None)  # type: ignore[arg-type]
    doc = DummyDocument({"source_file": "FDA_metformin_label.pdf"})

    enriched = ingestor.enrich_metadata([doc])[0]

    assert enriched.metadata["source_org"] == "FDA"
    assert enriched.metadata["evidence_type"] == "drug_label"
    assert enriched.metadata["specialty"] == "endocrinology"


def test_load_guideline_pdfs_respects_zero_limit(monkeypatch, tmp_path: Path):
    guideline_dir = tmp_path / "guidelines"
    guideline_dir.mkdir()
    (guideline_dir / "WHO_BP.pdf").write_bytes(b"fake pdf")
    monkeypatch.setenv("MAX_GUIDELINE_FILES", "0")

    ingestor = MedRAGIngestor(config=SimpleNamespace(data_dir=tmp_path))

    docs = ingestor._load_guideline_pdfs()

    assert docs == []


def test_load_and_parse_includes_bootstrap_documents(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUBMED_ENABLED", "false")
    monkeypatch.setenv("MEDRAG_INCLUDE_BOOTSTRAP", "true")

    ingestor = MedRAGIngestor(config=SimpleNamespace(data_dir=tmp_path))

    docs = ingestor.load_and_parse()

    titles = {doc.metadata["title"] for doc in docs}
    assert "Type 2 diabetes first-line therapy overview" in titles
    assert "Why medical guidance should include a disclaimer" in titles

def test_bootstrap_documents_are_split_into_sentence_level_documents():
    ingestor = MedRAGIngestor(config=None)  # type: ignore[arg-type]

    docs = ingestor._load_bootstrap_documents()

    type2_docs = [
        doc
        for doc in docs
        if doc.metadata["title"] == "Type 2 diabetes first-line therapy overview"
    ]

    assert len(type2_docs) == 2

    assert type2_docs[0].text == (
        "Type 2 diabetes management usually begins with lifestyle modification, "
        "glycemic monitoring, and metformin when there are no contraindications."
    )

    assert type2_docs[0].metadata["sentence_part"] == 1

    assert type2_docs[1].text == (
        "Guidelines commonly describe individualized escalation to GLP-1 receptor "
        "agonists, SGLT2 inhibitors, or insulin based on comorbidities, kidney "
        "function, cardiovascular risk, and glycemic control."
    )

    assert type2_docs[1].metadata["sentence_part"] == 2
