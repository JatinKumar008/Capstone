import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import load_and_chunk_documents
from src.index import build_dense_index, build_bm25_index
from src.pipeline import rag_pipeline

chunks = load_and_chunk_documents()
faiss_index, chunk_embeddings = build_dense_index(chunks)
bm25 = build_bm25_index(chunks)


def interactive_query(query: str):
    print("\n" + "="*60)
    print(f"❓ Query: {query}")
    print("-"*60)

    result = rag_pipeline(query, chunks, faiss_index, bm25, verbose=True)

    print(f"\n💬 Answer:")
    print(result["answer"])

    if result["retrieved_chunks"] and not result["refused"]:
        print(f"\n📎 Sources used:")
        for chunk in result["retrieved_chunks"]:
            print(f"  • [{chunk.chunk_id}] ({chunk.doc_type}): {chunk.text[:80]}...")

    print(f"\n⏱️ Latency: {result['latency_ms']}ms")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question to the RAG banking assistant")
    parser.add_argument("question", nargs="?", help="Your question (wrap in quotes)")
    args = parser.parse_args()

    if args.question:
        interactive_query(args.question)
    else:
        print("Usage: python scripts\\interactive.py \"your question here\"")
        while True:
            q = input("\nAsk a question (or type 'quit' to exit): ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                interactive_query(q)