from typing import List
from src.config import client, LLM_MODEL
from src.documents import Chunk

GENERATOR_SYSTEM_PROMPT = """
You are a strict and accurate banking assistant for NovaBank.
Answer using ONLY the provided context chunks — no outside knowledge.

STRICT RULES:
1. Every factual claim MUST be directly stated in context. Cite as [Source: chunk_id].
2. Do NOT infer, extrapolate, or generalize beyond the context.
3. If a specific value (rate, fee, limit) is absent from context, say:
   "This detail is not in our current documentation. Please contact your branch."
4. Never use 'typically', 'generally', 'usually' — only state what the context says.
5. Use bullet points for lists of fees or features.
6. If chunks conflict, cite both and flag the discrepancy.
"""

CONFIDENCE_THRESHOLD = 0.007

REFUSAL_MESSAGE = (
    "I'm sorry, I don't have reliable information to answer this question. "
    "Please contact your branch or our 24/7 helpline at 1800-XXX-XXXX for assistance."
)

OUT_OF_SCOPE_MESSAGE = (
    "This question is outside the scope of NovaBank's product and fee information. "
    "I can only assist with questions about our savings accounts, fixed deposits, "
    "personal loans, credit cards, fees, and eligibility criteria."
)


def check_retrieval_confidence(scores: List[float]) -> bool:
    if not scores:
        return False
    return scores[0] >= CONFIDENCE_THRESHOLD


def generate_answer(query: str, retrieved_chunks: List[Chunk], top_n: int = 3) -> str:
    context_str = "\n\n".join([
        f"[Source: {c.chunk_id}]\n{c.text}"
        for c in retrieved_chunks[:top_n]
    ])

    user_prompt = (
        f"Context (use ONLY this — no outside knowledge):\n{context_str}\n\n"
        f"Customer Question: {query}\n\n"
        f"Answer (cite every claim as [Source: chunk_id]):"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()