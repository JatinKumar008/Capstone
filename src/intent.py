import re
import time
from src.config import client, LLM_MODEL

OTHER_BANKS = re.compile(
    r"\b(ICICI|HDFC|SBI|Axis|Kotak|Yes\s*Bank|IndusInd|PNB|Bank\s*of\s*Baroda|"
    r"Canara|Union\s*Bank|HSBC|Citibank|Standard\s*Chartered)\b",
    re.IGNORECASE,
)

INTENT_SYSTEM_PROMPT = """
You are an intent classifier for a banking assistant.
Classify the user query into ONE of these intents:

- fees: Questions about charges, penalties, costs, service fees, transaction fees
- eligibility: Questions about who can apply, requirements, documents, criteria, qualifications
- product_terms: Questions about features, interest rates, tenure, limits, how products work
- out_of_scope: Questions unrelated to banking products (stocks, insurance, crypto, general advice)
- broad: Query spans multiple intents OR is ambiguous — search all document types

Respond with ONLY one word: fees | eligibility | product_terms | out_of_scope | broad
"""

INTENT_TO_DOC_TYPE = {
    "fees":          "fees",
    "eligibility":   "eligibility",
    "product_terms": "product_terms",
    "out_of_scope":  None,
    "broad":         None,
}

VALID_INTENTS = ["fees", "eligibility", "product_terms", "out_of_scope", "broad"]


def parse_intent(raw: str) -> str:
    text = raw.strip().lower().replace("out of scope", "out_of_scope")
    if not text:
        return "broad"
    best = (len(text) + 1, "broad")
    for token in VALID_INTENTS:
        m = re.search(rf"\b{re.escape(token)}\b", text)
        if m and m.start() < best[0]:
            best = (m.start(), token)
    return best[1]


def classify_intent(query: str) -> str:
    if OTHER_BANKS.search(query):
        return "out_of_scope"

    for attempt in range(3):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user",   "content": query}
            ],
            temperature=0.0,
            max_tokens=512
        )
        raw = response.choices[0].message.content
        cleaned = raw.strip().lower().replace("out of scope", "out_of_scope") if raw else ""
        if any(re.search(rf"\b{re.escape(t)}\b", cleaned) for t in VALID_INTENTS):
            return parse_intent(raw)
        if attempt < 2:
            print(f"  ⚠️  Empty/ambiguous reply (attempt {attempt+1}) → retrying")
    return "broad"