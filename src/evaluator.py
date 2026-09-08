import json
import re
import time
from src.config import client, LLM_MODEL
from src.documents import Chunk

FAITHFULNESS_PROMPT = """
You are evaluating whether an AI answer is faithful to the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Evaluate faithfulness: Does the answer ONLY contain claims supported by the context?
Score 0.0 to 1.0 where:
- 1.0 = all claims are grounded in context
- 0.5 = some claims unsupported
- 0.0 = answer contradicts or ignores context

Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}
"""

RELEVANCY_PROMPT = """
You are evaluating whether an AI answer addresses the user question.

Question: {question}
Answer: {answer}

Score 0.0 to 1.0 where:
- 1.0 = directly and completely answers the question
- 0.5 = partially answers or goes off-topic
- 0.0 = completely misses the question

Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}
"""


def extract_json(raw: str) -> dict:
    if not raw:
        return {}
    s = raw.strip()
    for _ in range(5):
        try:
            return json.loads(s)
        except Exception:
            pass
        if s.startswith("```"):
            s = s[3:].lstrip()
            if s.startswith("json"):
                s = s[4:].lstrip()
            continue
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end < start:
            return {}
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return {}


def llm_judge(prompt_template: str, **kwargs) -> dict:
    prompt = prompt_template.format(**kwargs)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content
    parsed = extract_json(raw)
    if not parsed:
        print(f"[DEBUG] llm_judge raw response: {repr(raw)}")
        return {"score": 0.0, "reason": "parse error"}
    return parsed


def evaluate_result(result: dict, ground_truth_answer: str = None,
                    should_be_refused: bool = False) -> dict:
    metrics = {}

    metrics["refusal_correctness"] = (
        1.0 if (should_be_refused == result["refused"]) else 0.0
    )

    if result["refused"]:
        metrics["faithfulness"] = None
        metrics["relevancy"] = None
        return metrics

    context = "".join([c.text for c in result["retrieved_chunks"]])

    faith = llm_judge(FAITHFULNESS_PROMPT, context=context,
                      question=result["query"], answer=result["answer"])
    metrics["faithfulness"] = faith.get("score", 0.0)
    metrics["faithfulness_reason"] = faith.get("reason", "")
    time.sleep(0.5)

    rel = llm_judge(RELEVANCY_PROMPT, question=result["query"], answer=result["answer"])
    metrics["relevancy"] = rel.get("score", 0.0)
    metrics["relevancy_reason"] = rel.get("reason", "")

    return metrics