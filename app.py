import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.documents import load_and_chunk_documents
from src.index import build_dense_index, build_bm25_index
from src.pipeline import rag_pipeline

st.set_page_config(page_title="NovaBank Assistant", page_icon="🏦", layout="centered")

st.title("🏦 NovaBank Assistant")
st.caption("Ask about NovaBank products, fees, and eligibility — grounded in official docs with cited sources.")


@st.cache_resource(show_spinner="Loading NovaBank knowledge base...")
def load_components():
    chunks = load_and_chunk_documents()
    faiss_index, chunk_embeddings = build_dense_index(chunks)
    bm25 = build_bm25_index(chunks)
    return chunks, faiss_index, bm25


chunks, faiss_index, bm25 = load_components()

EXAMPLE_QUESTIONS = [
    "What are the NEFT charges for NovaSavings account?",
    "What is the minimum CIBIL score for a NovaPersonal Loan?",
    "What is the interest rate on the NovaSavings Account?",
    "What documents do I need to apply for a NovaPersonal Loan?",
]

st.write("### Ask a question")
query = st.text_input("Your question", placeholder="e.g. What are the NEFT charges for NovaSavings account?")

col1, col2 = st.columns(2)
ask_clicked = col1.button("Ask", type="primary", use_container_width=True)
clear_clicked = col2.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.pop("last_result", None)
    st.rerun()

with st.expander("Try an example question"):
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=q, use_container_width=True):
            query = q
            ask_clicked = True

if ask_clicked and query.strip():
    with st.spinner("Thinking..."):
        result = rag_pipeline(query.strip(), chunks, faiss_index, bm25, verbose=False)

    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.markdown("### 💬 Answer")
    st.markdown(result["answer"])

    if result["refused"]:
        st.warning(f"Refused · reason: {result['refusal_reason']}")
    else:
        st.write("")
        st.markdown("### 📎 Sources used")
        for chunk, score in zip(result["retrieved_chunks"], result["rrf_scores"]):
            with st.expander(f"[{chunk.chunk_id}] ({chunk.doc_type}) — {score:.4f}"):
                st.write(chunk.text)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Intent", result["intent"])
    m2.metric("Latency", f"{result['latency_ms']:.0f} ms")
    m3.metric("Chunks retrieved", len(result["retrieved_chunks"]))
    m4.metric("Top RRF", f"{result['rrf_scores'][0]:.4f}" if result["rrf_scores"] else "—")