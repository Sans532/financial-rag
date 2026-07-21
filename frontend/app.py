"""Streamlit frontend for the Financial RAG system. Talks to the FastAPI backend over HTTP."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial RAG", layout="wide")
st.title("Financial Filing RAG Assistant")
st.caption("Multi-agent RAG over SEC filings, XBRL ground truth, and market data.")

with st.sidebar:
    st.header("Query options")
    companies_input = st.text_input("Tickers (comma-separated, optional)", value="")
    max_retries = st.slider("Verifier max retries", 0, 5, 2)
    st.divider()
    st.markdown(
        "**Example questions**\n\n"
        "- How did Apple's gross margin change last quarter and why?\n"
        "- Compare JPMorgan and Bank of America's net income YoY.\n"
        "- What risk factors did NVIDIA highlight in its most recent 10-K?\n"
        "- What was Pfizer's R&D expense in its latest 10-Q?"
    )

question = st.text_area("Ask a question about a covered company", height=100)

if st.button("Ask", type="primary") and question.strip():
    companies = [c.strip().upper() for c in companies_input.split(",") if c.strip()]
    with st.spinner("Running planner -> retrieval -> synthesis -> verification..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "companies": companies, "max_retries": max_retries},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(data["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Faithfulness score", f"{data['faithfulness_score']:.0%}")
    col2.metric("Latency", f"{data['total_latency_ms'] / 1000:.1f}s")
    col3.metric("Citations", len(data["citations"]))

    if data["citations"]:
        st.subheader("Citations")
        st.dataframe(data["citations"], use_container_width=True)

    if data["verification"]:
        st.subheader("Numeric claim verification")
        st.dataframe(data["verification"], use_container_width=True)

    with st.expander("Agent trace"):
        st.dataframe(data["trace"], use_container_width=True)

    if data["errors"]:
        error_lines = "\n".join(f"- {e}" for e in data["errors"])
        st.warning(f"Non-fatal errors during this run:\n{error_lines}")

st.divider()
if st.button("Load latest eval report"):
    try:
        resp = requests.get(f"{API_URL}/eval-report", timeout=10)
        resp.raise_for_status()
        st.markdown(resp.json()["report_markdown"])
    except requests.RequestException as e:
        st.info(f"No eval report available yet ({e}). Run `python scripts/run_eval.py`.")
