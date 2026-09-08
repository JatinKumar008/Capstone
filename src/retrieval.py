import numpy as np
from typing import List, Optional
from src.index import embed_model, tokenize
from src.documents import Chunk


def dense_search(chunks: List[Chunk], faiss_index, query: str, top_k: int = 20,
                 doc_type_filter: Optional[str] = None) -> List[tuple]:
    query_vec = embed_model.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = faiss_index.search(query_vec, top_k * 3)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        if doc_type_filter and chunks[idx].doc_type != doc_type_filter:
            continue
        results.append((idx, float(score)))
        if len(results) == top_k:
            break
    return results


def sparse_search(chunks: List[Chunk], bm25, query: str, top_k: int = 20,
                  doc_type_filter: Optional[str] = None) -> List[tuple]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices:
        if doc_type_filter and chunks[idx].doc_type != doc_type_filter:
            continue
        results.append((idx, float(scores[idx])))
        if len(results) == top_k:
            break
    return results


def reciprocal_rank_fusion(dense_results: List[tuple], sparse_results: List[tuple],
                           k: int = 60, top_n: int = 5) -> List[tuple]:
    rrf_scores = {}
    for rank, (idx, _) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(sparse_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


PRODUCTS = [
    "novaprime", "novasavings", "nova fixed deposit", "novafd",
    "nova recurring deposit", "novard", "nova personal loan",
    "nova credit card", "nova creditcard", "nova bankk", "novabank"
]


def detect_product(query: str) -> Optional[str]:
    q = query.lower()
    for p in PRODUCTS:
        if p in q:
            return p
    return None


def promote_matching_product(fused: List[tuple], chunks: List[Chunk], query: str) -> List[tuple]:
    product = detect_product(query)
    if product is None:
        return fused

    def rank_key(item):
        idx, _ = item
        return 0 if product in chunks[idx].text.lower() else 1

    return sorted(fused, key=rank_key)


def hybrid_retrieve(chunks: List[Chunk], faiss_index, bm25, query: str,
                    top_n: int = 5, doc_type_filter: Optional[str] = None) -> List[Chunk]:
    dense_res = dense_search(chunks, faiss_index, query, top_k=20, doc_type_filter=doc_type_filter)
    sparse_res = sparse_search(chunks, bm25, query, top_k=20, doc_type_filter=doc_type_filter)
    fused = reciprocal_rank_fusion(dense_res, sparse_res, top_n=top_n * 2)
    fused = promote_matching_product(fused, chunks, query)
    fused = fused[:top_n]
    return [chunks[idx] for idx, _ in fused], [score for _, score in fused]