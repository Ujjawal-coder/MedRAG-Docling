from __future__ import annotations

import re
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from src.core.base import DocumentIngestor, SourceDocument


class MedRAGIngestor(DocumentIngestor):
    bootstrap_documents = (
        {
            "title": "Type 2 diabetes first-line therapy overview",
            "text": (
                "Type 2 diabetes management usually begins with lifestyle modification, "
                "glycemic monitoring, and metformin when there are no contraindications. "
                "Guidelines commonly describe individualized escalation to GLP-1 receptor "
                "agonists, SGLT2 inhibitors, or insulin based on comorbidities, kidney "
                "function, cardiovascular risk, and glycemic control."
            ),
            "specialty": "endocrinology",
        },
        {
            "title": "Type 1 diabetes treatment overview",
            "text": (
                "Type 1 diabetes requires insulin replacement therapy rather than oral "
                "first-line agents. Guideline-based care typically includes basal-bolus "
                "or pump-based insulin delivery, glucose monitoring, hypoglycemia "
                "education, and individualized nutrition planning."
            ),
            "specialty": "endocrinology",
        },
        {
            "title": "Hypertension guideline overview",
            "text": (
                "Hypertension guidelines generally cover blood pressure thresholds, "
                "target ranges, home blood pressure monitoring, lifestyle changes, and "
                "medication classes such as ACE inhibitors, ARBs, calcium-channel "
                "blockers, and thiazide-type diuretics. Follow-up cadence depends on "
                "severity, symptoms, and treatment response."
            ),
            "specialty": "cardiology",
        },
        {
            "title": "Why medical guidance should include a disclaimer",
            "text": (
                "Medical guidance in retrieval-augmented systems should include a disclaimer "
                "that the information is for educational purposes only and is not a substitute "
                "for clinician judgment, diagnosis, or personalized treatment. The disclaimer "
                "helps users understand the limits of the retrieved material and encourages "
                "consultation with qualified healthcare professionals."
            ),
            "specialty": "general_medicine",
        },
    )

    pubmed_queries = (
        "type 2 diabetes treatment guideline",
        "hypertension management guideline",
    )
    pubmed_max_results = 5

    def load_and_parse(self):
        documents = []
        documents.extend(self._load_guideline_pdfs())
        documents.extend(self._load_pubmed_abstracts())

        if os.getenv("MEDRAG_INCLUDE_BOOTSTRAP", "true").lower() in {
            "1",
            "true",
            "yes",
        }:
            documents.extend(self._load_bootstrap_documents())

        return documents

    def enrich_metadata(self, docs):
        for doc in docs:
            metadata = dict(doc.metadata or {})
            source_file = str(metadata.get("source_file", "")).lower()

            text_hint = " ".join(
                [
                    source_file,
                    str(metadata.get("title", "")),
                    str(metadata.get("query", "")),
                ]
            ).lower()

            if metadata.get("source") == "pubmed":
                metadata["source_org"] = "PubMed"
                metadata["evidence_type"] = "research_abstract"
            elif any(token in source_file for token in ("fda", "dailymed", "label")):
                metadata["source_org"] = "FDA"
                metadata["evidence_type"] = "drug_label"
            else:
                metadata["source_org"] = metadata.get("source_org", "WHO")
                metadata["evidence_type"] = metadata.get(
                    "evidence_type",
                    "guideline",
                )

            metadata["specialty"] = self._infer_specialty(text_hint)
            doc.metadata = metadata

        return docs

    def _load_guideline_pdfs(self):
        guideline_dir = Path(self.config.data_dir) / "guidelines"
        pdf_paths = sorted(guideline_dir.glob("*.pdf"))

        max_guideline_files = int(os.getenv("MAX_GUIDELINE_FILES", "3"))
        if max_guideline_files >= 0:
            pdf_paths = pdf_paths[:max_guideline_files]

        if not pdf_paths:
            return []

        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        documents = []

        for pdf_path in pdf_paths:
            result = converter.convert(source=pdf_path)
            docling_document = result.document

            documents.append(
                SourceDocument(
                    text=docling_document.export_to_markdown(),
                    metadata={
                        "source_file": pdf_path.name,
                        "parser": "docling",
                        "source": "guideline_pdf",
                    },
                    docling_document=docling_document,
                )
            )

        return documents

    def _load_pubmed_abstracts(self):
        if os.getenv("PUBMED_ENABLED", "true").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return []

        documents = []

        max_queries = int(os.getenv("PUBMED_QUERY_LIMIT", "1"))
        queries = self.pubmed_queries[:max_queries] if max_queries > 0 else ()

        max_results = int(
            os.getenv("PUBMED_MAX_RESULTS", str(self.pubmed_max_results))
        )

        for query in queries:
            pmids = self._search_pubmed(query, max_results)

            if not pmids:
                continue

            for article in self._fetch_pubmed_articles(pmids):
                title = article["title"]
                abstract = article["abstract"]

                if not abstract:
                    continue

                text = f"{title}\n\n{abstract}" if title else abstract

                documents.append(
                    SourceDocument(
                        text=text,
                        metadata={
                            "source": "pubmed",
                            "source_file": f"pubmed:{article['pmid']}",
                            "title": title,
                            "query": query,
                            "pmid": article["pmid"],
                            "parser": "pubmed_api",
                        },
                    )
                )

        return documents
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]
    @staticmethod
    def _search_pubmed(query: str, max_results: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "tool": "medrag_toolkit",
        }

        MedRAGIngestor._add_ncbi_credentials(params)

        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        return response.json().get("esearchresult", {}).get("idlist", [])

    @staticmethod
    def _fetch_pubmed_articles(pmids: list[str]) -> list[dict[str, str]]:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "medrag_toolkit",
        }

        MedRAGIngestor._add_ncbi_credentials(params)

        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        articles = []

        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//MedlineCitation/PMID")
            title_node = article.find(".//Article/ArticleTitle")

            pmid = (
                "".join(pmid_node.itertext()).strip()
                if pmid_node is not None
                else ""
            )
            title = (
                "".join(title_node.itertext()).strip()
                if title_node is not None
                else ""
            )

            abstract_parts = []

            for abstract_node in article.findall(
                ".//Article/Abstract/AbstractText"
            ):
                text = "".join(abstract_node.itertext()).strip()

                if not text:
                    continue

                label = abstract_node.attrib.get("Label")
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": "\n".join(abstract_parts),
                }
            )

        return articles

    @staticmethod
    def _add_ncbi_credentials(params: dict) -> None:
        email = os.getenv("NCBI_EMAIL")
        api_key = os.getenv("NCBI_API_KEY")

        if email:
            params["email"] = email

        if api_key:
            params["api_key"] = api_key

    def _load_bootstrap_documents(self):
        documents = []

        for item in self.bootstrap_documents:
            parts = self._split_sentences(item["text"])
            for part_number, text in enumerate(parts, start=1):
                documents.append(
                        SourceDocument(
                            text=text,
                            metadata={
                                "source": "bootstrap",
                                "source_file": "bootstrap_seed",
                                "title": item["title"],
                                "parser": "bootstrap_seed",
                                "source_org": "Bootstrap",
                                "evidence_type": "guideline_summary",
                                "specialty": item["specialty"],
                                "sentence_part": part_number,
                    },
                )
            )

        return documents

    @staticmethod
    def _infer_specialty(text_hint: str) -> str:
        mapping = {
            "diabetes": "endocrinology",
            "glucose": "endocrinology",
            "metformin": "endocrinology",
            "insulin": "endocrinology",
            "hypertension": "cardiology",
            "blood pressure": "cardiology",
            "warfarin": "cardiology",
            "lisinopril": "cardiology",
            "asthma": "pulmonology",
            "copd": "pulmonology",
            "kidney": "nephrology",
        }

        for keyword, specialty in mapping.items():
            if keyword in text_hint:
                return specialty

        return "general_medicine"