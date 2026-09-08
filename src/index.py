import numpy as np
import faiss
import re
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.documents import Chunk

print("⏳ Loading BGE embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✅ Embedding model loaded")


def build_dense_index(chunks: List[Chunk]):
    texts = [c.text for c in chunks]
    print(f"⏳ Encoding {len(texts)} chunks...")

    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"✅ FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


def build_bm25_index(chunks: List[Chunk]):
    corpus_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"✅ BM25 index built over {len(corpus_tokens)} documents")
    return bm25