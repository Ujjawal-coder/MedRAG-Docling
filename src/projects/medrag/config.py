from __future__ import annotations

from src.core.base import MetadataField, ProjectConfig
from src.core.settings import project_root


ROOT = project_root()

MEDRAG_CONFIG = ProjectConfig(
    name="medrag",
    collection_name="medrag_collection_bge_small",
    system_prompt=(
        "You are a clinical guidelines assistant. Answer medical questions using only the "
        "retrieved context. Cite source organizations when possible, state uncertainty when "
        "the context is incomplete, and do not provide personalized medical advice."
    ),
    disclaimer="For educational purposes only. This is not medical advice.",
    data_dir=ROOT / "src/projects/medrag/data",
    metadata_fields=[
        MetadataField(
            name="source_org",
            type="str",
            description="Publishing organization such as FDA, WHO, CDC, AHA, or PubMed",
        ),
        MetadataField(
            name="specialty",
            type="str",
            description="Clinical specialty inferred from the document topic",
        ),
        MetadataField(
            name="evidence_type",
            type="str",
            description="Document category: guideline, drug_label, or research_abstract",
        ),
    ],
    golden_dataset_path=ROOT / "eval/medrag/golden_dataset.json",
)
