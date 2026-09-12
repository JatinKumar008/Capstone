import os
import sys
import re
import time
import statistics
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("⏳ Initializing pipeline components...")
from src.documents import load_and_chunk_documents
from src.index import build_dense_index, build_bm25_index
from src.pipeline import rag_pipeline
from src.evaluator import evaluate_result

chunks = load_and_chunk_documents()
faiss_index, chunk_embeddings = build_dense_index(chunks)
bm25 = build_bm25_index(chunks)

qa_set = []
with open("data/qa_evaluation_set.txt", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        m = re.match(
            r"^(?:\d+\.\s*)?Q:\s*(.*?)\s*\|\s*A:\s*(.*?)\s*(?:\|\s*answerable:\s*(no|yes))?\s*$",
            line
        )
        if m:
            flag = (m.group(3) or "yes").lower()
            qa_set.append({
                "question": m.group(1),
                "answer": m.group(2),
                "answerable": flag != "no",
            })

EVAL_SAMPLE_SIZE = 70
eval_sample = qa_set[:EVAL_SAMPLE_SIZE]

print(f"\n📊 Running evaluation on {min(EVAL_SAMPLE_SIZE, len(qa_set))} samples...")
print("(This will take a while due to API calls)\n")

eval_results = []
latencies = []

for i, qa in enumerate(eval_sample):
    print(f"  [{i+1}/{EVAL_SAMPLE_SIZE}] {qa['question'][:200]}...")

    result = rag_pipeline(qa["question"], chunks, faiss_index, bm25, verbose=False)
    latencies.append(result["latency_ms"])

    should_refuse = not qa.get("answerable", True)
    metrics = evaluate_result(result, ground_truth_answer=qa.get("answer", ""),
                              should_be_refused=should_refuse)

    eval_results.append({
        "question": qa["question"],
        "expected_answer": qa.get("answer", ""),
        "generated_answer": result["answer"],
        "intent": result["intent"],
        "refused": result["refused"],
        "should_refuse": should_refuse,
        "latency_ms": result["latency_ms"],
        **metrics
    })

    time.sleep(2)

print("\n✅ Evaluation complete!")

answered = [r for r in eval_results if not r["refused"]]
refused_results = [r for r in eval_results if r["refused"]]

def _mean(values):
    return statistics.mean(values) if values else 0

faith_scores = [r["faithfulness"] for r in answered if r.get("faithfulness") is not None]
avg_faithfulness = _mean(faith_scores)

correct_scores = [r["correctness"] for r in answered if r.get("correctness") is not None]
avg_correctness = _mean(correct_scores)

refusal_correct = [r for r in eval_results if r["refusal_correctness"] == 1.0]
refusal_correctness = len(refusal_correct) / len(eval_results) if eval_results else 0

sorted_latencies = sorted(latencies)
p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
avg_latency = statistics.mean(latencies) if latencies else 0

print("\n" + "="*50)
print("📊 EVALUATION RESULTS")
print("="*50)
print(f"  Samples evaluated    : {len(eval_results)}")
print(f"  Answered             : {len(answered)}")
print(f"  Refused              : {len(refused_results)}")
print(f"")
print(f"  Golden Accuracy      : {avg_correctness:.3f}")
print(f"  Faithfulness         : {avg_faithfulness:.3f}")
print(f"  Refusal Correctness  : {refusal_correctness:.3f}")
print(f"")
print(f"  Avg Latency          : {avg_latency:.0f}ms")
print(f"  p95 Latency          : {p95_latency:.0f}ms")
print("="*50)

with open("data/eval/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2, default=str)

summary = {
    "samples": len(eval_results),
    "accuracy": round(avg_correctness, 3),
    "faithfulness": round(avg_faithfulness, 3),
    "refusal_correctness": round(refusal_correctness, 3),
    "p95_latency_ms": round(p95_latency, 1)
}
with open("data/eval/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Results saved to data/eval/")

low_faith = [
    r for r in answered
    if r.get("faithfulness") is not None and r["faithfulness"] < 0.7
]

low_acc = [
    r for r in answered
    if r.get("correctness") is not None and r["correctness"] < 0.7
]

wrong_refusals = [r for r in eval_results if r["refusal_correctness"] == 0.0]

print(f"\n⚠️  Low Faithfulness Cases (< 0.7): {len(low_faith)}")
for r in low_faith[:3]:
    print(f"  Q: {r['question'][:80]}")
    print(f"  Faith: {r['faithfulness']:.2f} | {r.get('faithfulness_reason', '')}")
    print()

print(f"\n⚠️  Low Accuracy Cases vs Golden (< 0.7): {len(low_acc)}")
for r in low_acc[:3]:
    print(f"  Q: {r['question'][:80]}")
    print(f"  Acc: {r['correctness']:.2f} | {r.get('correctness_reason', '')}")
    print()

print(f"\n⚠️  Wrong Refusal Decisions: {len(wrong_refusals)}")
for r in wrong_refusals[:3]:
    refused_when = "should NOT have" if not r["should_refuse"] else "should have"
    print(f"  Q: {r['question'][:80]}")
    print(f"  System refused={r['refused']} but {refused_when} refused")
    print()