import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.documents import load_and_chunk_documents
from src.index import build_dense_index, build_bm25_index
from src.pipeline import rag_pipeline
from src.memory import ConversationMemory

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

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLE_QUESTIONS = [
    "What are the NEFT charges for NovaSavings account?",
    "What is the minimum CIBIL score for a NovaPersonal Loan?",
    "What is the interest rate on the NovaSavings Account?",
    "What documents do I need to apply for a NovaPersonal Loan?",
]


def ask_question(q: str):
    with st.spinner("Thinking..."):
        result = rag_pipeline(
            q, chunks, faiss_index, bm25, verbose=False,
            conversation=st.session_state.memory.load_memory(),
        )
    st.session_state.memory.save_context(q, result["answer"])
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})


st.sidebar.markdown("### 💬 Conversation controls")

if st.sidebar.button("New conversation", use_container_width=True):
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Example questions**")
for q in EXAMPLE_QUESTIONS:
    if st.sidebar.button(q, key=q, use_container_width=True):
        ask_question(q)

with st.sidebar.expander("🧠 Memory inspector", expanded=False):
    memory = st.session_state.memory
    buffered = len(memory.buffer)
    st.write(f"**Buffer turns:** {buffered}")
    st.write(f"**Summarized:** {'yes' if memory.summary else 'no'}")
    st.write(f"**Estimated tokens fed to LLM:** {memory._approx_tokens(memory.load_memory())}")
    st.divider()
    st.caption("**What gets passed as 'Previous conversation':**")
    st.code(memory.load_memory() or "(empty — memory not being tracked)", language="text")

prompt = st.chat_input("Ask about NovaBank products, fees, and eligibility…")
if prompt and prompt.strip():
    ask_question(prompt.strip())

st.write("### 💬 Conversation")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("result"):
            result = msg["result"]
            if result["refused"]:
                st.warning(f"Refused · reason: {result['refusal_reason']}")
            else:
                with st.expander("📎 Sources used"):
                    for chunk, score in zip(result["retrieved_chunks"], result["rrf_scores"]):
                        st.write(f"[{chunk.chunk_id}] ({chunk.doc_type}) — {score:.4f}")
                        st.caption(chunk.text)
            m1, m2 = st.columns(2)
            m1.caption(f"Intent: {result['intent']}")
            m2.caption(f"Latency: {result['latency_ms']:.0f} ms")