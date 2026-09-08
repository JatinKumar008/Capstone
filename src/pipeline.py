import time
from typing import List
from src.documents import Chunk
from src.retrieval import hybrid_retrieve
from src.intent import classify_intent, INTENT_TO_DOC_TYPE
from src.generator import (
    check_retrieval_confidence, generate_answer,
    REFUSAL_MESSAGE, OUT_OF_SCOPE_MESSAGE
)


def rag_pipeline(query: str, chunks, faiss_index, bm25,
                 top_n: int = 5, verbose: bool = True) -> dict:
    start = time.time()

    intent = classify_intent(query)
    if verbose:
        print(f"  🎯 Intent: {intent}")

    if intent == "out_of_scope":
        return {
            "query": query, "answer": OUT_OF_SCOPE_MESSAGE,
            "intent": intent, "retrieved_chunks": [], "rrf_scores": [],
            "refused": True, "refusal_reason": "out_of_scope",
            "latency_ms": round((time.time() - start) * 1000, 2)
        }

    doc_filter = INTENT_TO_DOC_TYPE.get(intent)
    if verbose and doc_filter is None:
        print(f"  ℹ️  Broad retrieval (no namespace filter)")

    retrieved, scores = hybrid_retrieve(chunks, faiss_index, bm25, query,
                                        top_n=top_n, doc_type_filter=doc_filter)
    if verbose:
        print(f"  🔍 Retrieved {len(retrieved)} chunks (top RRF: {scores[0]:.4f})")

    if not check_retrieval_confidence(scores):
        return {
            "query": query, "answer": REFUSAL_MESSAGE,
            "intent": intent, "retrieved_chunks": retrieved, "rrf_scores": scores,
            "refused": True, "refusal_reason": "low_confidence",
            "latency_ms": round((time.time() - start) * 1000, 2)
        }

    answer = generate_answer(query, retrieved, top_n=3)

    return {
        "query": query, "answer": answer, "intent": intent,
        "retrieved_chunks": retrieved, "rrf_scores": scores,
        "refused": False, "refusal_reason": None,
        "latency_ms": round((time.time() - start) * 1000, 2)
    }