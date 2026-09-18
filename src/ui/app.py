from __future__ import annotations

import os
from urllib.parse import quote

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
QUERY_TIMEOUT_SECONDS = 120
REINDEX_TIMEOUT_SECONDS = 600
EVAL_TIMEOUT_SECONDS = 900


def _get_health() -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _get_sources() -> tuple[dict | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}/sources", timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _reindex_sources() -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/sources/reindex",
            timeout=REINDEX_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _get_latest_eval_results() -> tuple[dict | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}/evals/medrag/latest", timeout=10)
        if response.status_code == 404:
            return None, None
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _run_eval() -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/evals/medrag/run",
            timeout=EVAL_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _render_source_table(sources: list[dict]) -> None:
    if not sources:
        st.info("No uploaded PDF sources yet.")
        return
    st.dataframe(
        [
            {
                "Name": source["name"],
                "Size": _format_size(source["size_bytes"]),
                "Updated (UTC)": source["modified_at"],
            }
            for source in sources
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_pubmed_status(pubmed: dict) -> None:
    st.subheader("PubMed Ingestion Status")
    status_label = "Enabled" if pubmed.get("enabled") else "Disabled"
    metric_cols = st.columns(4)
    metric_cols[0].metric("Status", status_label)
    metric_cols[1].metric("Query Limit", str(pubmed.get("configured_query_limit", 0)))
    metric_cols[2].metric("Max Results", str(pubmed.get("configured_max_results", 0)))
    metric_cols[3].metric("Indexed Docs", str(pubmed.get("indexed_document_count", 0)))

    configured_queries = pubmed.get("configured_queries", [])
    if configured_queries:
        st.caption("Configured PubMed queries")
        for query in configured_queries:
            st.write(f"- {query}")
    else:
        st.caption("No PubMed queries are configured for this project.")

    st.subheader("Last Indexed PubMed Queries")
    summaries = pubmed.get("indexed_query_summaries", [])
    if not summaries:
        st.info("No PubMed documents are currently reflected in the indexed collection.")
        return
    st.dataframe(
        [
            {
                "Query": summary["query"],
                "Document Count": summary["document_count"],
                "Chunk Count": summary["chunk_count"],
            }
            for summary in summaries
        ],
        use_container_width=True,
        hide_index=True,
    )


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _render_eval_results(payload: dict) -> None:
    summary = payload["summary"]
    status_label = "pass" if summary["success"] else "fail"
    metric_cols = st.columns(5)
    metric_cols[0].metric("Status", status_label)
    metric_cols[1].metric("Cases", str(summary["dataset_size"]))
    metric_cols[2].metric("Passed", str(summary["passed_cases"]))
    metric_cols[3].metric("Failed", str(summary["failed_cases"]))
    metric_cols[4].metric("Duration", f"{summary['duration_seconds']:.2f}s")
    st.caption(
        "Latest run completed at "
        f"{summary['completed_at']} for `{summary['collection_name']}`. "
        f"Success rate: {summary['success_rate'] * 100:.1f}%."
    )

    case_rows = [
        {
            "Case": case["id"],
            "Query": case["query"],
            "Status": "pass" if case["success"] else "fail",
            "Metrics Passed": f"{sum(1 for metric in case['metrics'] if metric['success'])}/{len(case['metrics'])}",
        }
        for case in payload["cases"]
    ]
    st.subheader("Case Overview")
    st.dataframe(case_rows, use_container_width=True, hide_index=True)

    st.subheader("Case Details")
    for case in payload["cases"]:
        status_label = "PASS" if case["success"] else "FAIL"
        with st.expander(f"{status_label} · {case['id']} · {case['query']}"):
            st.caption("Expected answer")
            st.write(case["expected_answer"])

            st.caption("Actual answer")
            st.write(case["actual_answer"])

            st.caption("Sources")
            for source in case["sources"]:
                st.write(f"- {source}")

            st.caption("Retrieved context")
            for snippet in case["retrieval_context"]:
                st.write(f"- {snippet}")

            st.caption("Metric results")
            st.dataframe(
                [
                    {
                        "Metric": metric["name"],
                        "Score": _format_score(metric["score"]),
                        "Threshold": _format_score(metric["threshold"]),
                        "Status": "pass" if metric["success"] else "fail",
                        "Reason": metric["reason"] or "",
                        "Error": metric["error"] or "",
                    }
                    for metric in case["metrics"]
                ],
                use_container_width=True,
                hide_index=True,
            )

st.set_page_config(page_title="MedRAG", page_icon="M", layout="wide")
st.title("MedRAG")
st.caption("Clinical guideline Q&A powered by a shared RAG pipeline.")

health_payload = _get_health()
if health_payload:
    status = "ready" if health_payload["collection_ready"] else "not indexed"
    st.write(
        "Project: "
        f"`{health_payload['project']}` | "
        f"Collection: `{health_payload['collection_name']}` | "
        f"Status: `{status}`"
    )
else:
    st.warning("API is not reachable yet. Start the FastAPI app before using the UI.")

query_tab, sources_tab, eval_tab = st.tabs(["Ask Questions", "Uploaded Sources", "Eval Results"])

with query_tab:
    question = st.text_area(
        "Ask a clinical question",
        placeholder="What do the guidelines say about first-line treatment for type 2 diabetes?",
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching the knowledge base and drafting an answer..."):
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"question": question},
                        timeout=QUERY_TIMEOUT_SECONDS,
                    )
            except requests.ReadTimeout:
                st.error(
                    "The request timed out. The first query after startup can be slower while the "
                    "local embedding model warms up. Please try again once."
                )
            except requests.RequestException as exc:
                st.error(f"Query failed: {exc}")
            else:
                if response.ok:
                    data = response.json()
                    st.subheader("Answer")
                    st.write(data["answer"])

                    st.subheader("Evidence")
                    st.write(data["evidence"])

                    st.subheader("Sources")
                    for source in data["sources"]:
                        st.write(f"- {source}")

                    st.caption(f"Confidence: {data['confidence']}")
                    st.info(data["disclaimer"])
                else:
                    st.error(response.text)

with sources_tab:
    st.caption(
        "Upload and manage local PDF sources for the active MedRAG collection. "
        "PubMed documents are controlled separately through environment settings."
    )

    uploaded_files = st.file_uploader(
        "Add guideline PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded files are stored in the MedRAG data directory and included on reindex.",
    )
    if st.button("Upload and reindex", disabled=not uploaded_files):
        try:
            with st.spinner("Uploading PDFs and rebuilding the index..."):
                for uploaded_file in uploaded_files or []:
                    response = requests.post(
                        f"{API_BASE_URL}/sources/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                _, error = _reindex_sources()
                if error:
                    st.error(f"Upload succeeded, but reindex failed: {error}")
                else:
                    st.success("Uploaded sources and rebuilt the collection.")
                    st.rerun()
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    source_payload, sources_error = _get_sources()
    if sources_error:
        st.error(f"Could not load uploaded sources: {sources_error}")
    else:
        sources = source_payload.get("sources", []) if source_payload else []
        pubmed = source_payload.get("pubmed", {}) if source_payload else {}
        st.subheader("Current Sources")
        _render_source_table(sources)
        _render_pubmed_status(pubmed)

        if sources:
            selected_source = st.selectbox(
                "Delete a source",
                options=[source["name"] for source in sources],
                index=None,
                placeholder="Choose a PDF to remove",
            )
            if st.button("Delete selected source", disabled=selected_source is None):
                try:
                    with st.spinner("Deleting the source and rebuilding the index..."):
                        response = requests.delete(
                            f"{API_BASE_URL}/sources/{quote(selected_source or '', safe='')}",
                            timeout=30,
                        )
                        response.raise_for_status()
                        _, error = _reindex_sources()
                        if error:
                            st.error(f"Source deleted, but reindex failed: {error}")
                        else:
                            st.success(f"Deleted '{selected_source}' and rebuilt the collection.")
                            st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Delete failed: {exc}")

        if st.button("Rebuild index from current sources"):
            with st.spinner("Rebuilding the collection from current sources..."):
                result, error = _reindex_sources()
            if error:
                st.error(f"Reindex failed: {error}")
            else:
                st.success(
                    f"{result['message']} Indexed {result['indexed_documents']} documents."
                )

with eval_tab:
    st.caption(
        "Run the local MedRAG golden eval from the dashboard and inspect the latest results "
        "without leaving Streamlit."
    )

    latest_eval_payload, latest_eval_error = _get_latest_eval_results()
    run_eval_error: str | None = None

    if st.button("Run MedRAG eval", type="primary"):
        with st.spinner("Running the MedRAG golden eval..."):
            run_payload, run_eval_error = _run_eval()
        if run_eval_error:
            st.error(f"Eval failed: {run_eval_error}")
        else:
            latest_eval_payload = run_payload
            st.success("Eval completed and the latest local results were updated.")

    if latest_eval_error:
        st.error(f"Could not load local eval results: {latest_eval_error}")
    elif latest_eval_payload is None:
        st.info("No local eval results yet. Run the MedRAG eval to populate this page.")
    else:
        _render_eval_results(latest_eval_payload)
