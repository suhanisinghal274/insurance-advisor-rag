"""Evaluate the assistant on a fixed test set.

Metrics (one row per question is saved to eval/results/results.csv):
- Retrieval recall@k: the expected policy section is among the top-k retrieved sections
- MRR: mean reciprocal rank of the expected section
- Answer accuracy: answerable questions answered (not escalated) and containing every expected keyword,
  after normalising text (Unicode spaces such as U+202F, "25 %" -> "25%", markdown bold, case)
- Escalation precision / recall: how well it hands off questions the documents can't answer
- Task completion: correct answer OR correct hand-off, over all questions
- Latency: p50 and p95 per question, in milliseconds

Run:  python -m src.evaluate
"""
import csv
import json
import re
import statistics
import time
import unicodedata

from src import config
from src.rag import AdvisorAssistant, QuotaExhausted


def normalise(text):
    """Make keyword matching robust to formatting, not wording: models often write "25 %" with a
    narrow no-break space, or wrap numbers in **bold**. Different wording still counts as a miss."""
    text = unicodedata.normalize("NFKC", text).replace("*", "").lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"(\d) %", r"\1%", text)


def percentile(values, p):
    values = sorted(values)
    idx = min(len(values) - 1, round(p / 100 * (len(values) - 1)))
    return values[idx]


def run():
    tests = json.loads(config.TEST_SET.read_text(encoding="utf-8"))
    bot = AdvisorAssistant(persist=False)

    # Answers are cached per model, so a run stopped by a daily quota resumes where it left off
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # The cache key includes the settings that change which answers the model is asked for
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", bot.llm.model)  # e.g. openai/gpt-oss-120b -> openai_gpt-oss-120b
    cache_path = config.RESULTS_DIR / f"answers_{safe_model}_k{config.TOP_K}_t{config.ESCALATE_BELOW}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache:
        print(f"Resuming: {len(cache)}/{len(tests)} answers already saved for {bot.llm.model}")

    calls = 0
    for t in tests:
        if t["id"] in cache:
            continue
        if calls and config.LLM_BACKEND != "mock":
            time.sleep(config.EVAL_DELAY_S)  # stay under free-tier per-minute limits (not counted in latency)
        try:
            cache[t["id"]] = bot.answer(t["question"])
        except QuotaExhausted as e:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            print(f"\nStopped: {e}\nSaved {len(cache)}/{len(tests)} answers to {cache_path.name}. "
                  "Run this again after the quota resets to finish; saved answers won't be re-sent.")
            return None
        calls += 1
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    rows = []
    for t in tests:
        out = cache[t["id"]]
        ids = [s["id"] for s in out["sources"]]
        row = {"id": t["id"], "question": t["question"], "answerable": t["answerable"],
               "escalated": out["escalated"], "top_score": out["top_score"],
               "latency_ms": out["latency_ms"], "answer": out["answer"]}
        if t["answerable"]:
            exp = t["expected_section"]
            row["retrieved"] = exp in ids
            row["rr"] = 1 / (ids.index(exp) + 1) if exp in ids else 0.0
            row["correct"] = (not out["escalated"]) and all(normalise(k) in normalise(out["answer"]) for k in t["keywords"])
        else:
            row["retrieved"], row["rr"] = None, None
            row["correct"] = out["escalated"]
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    true_esc = sum(r["escalated"] for r in unans)
    all_esc = sum(r["escalated"] for r in rows)
    lat = [r["latency_ms"] for r in rows]

    metrics = {
        "questions": len(rows),
        "retrieval_recall_at_k": sum(r["retrieved"] for r in ans) / len(ans),
        "mrr": statistics.mean(r["rr"] for r in ans),
        "answer_accuracy": sum(r["correct"] for r in ans) / len(ans),
        "escalation_precision": true_esc / all_esc if all_esc else 0.0,
        "escalation_recall": true_esc / len(unans) if unans else 0.0,
        "task_completion": sum(r["correct"] for r in rows) / len(rows),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "settings": {"embeddings": config.EMBEDDING_BACKEND, "llm": config.LLM_BACKEND,
                     "model": bot.llm.model,
                     "top_k": config.TOP_K, "escalate_below": config.ESCALATE_BELOW},
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nEvaluated {len(rows)} questions  (k={config.TOP_K}, threshold={config.ESCALATE_BELOW})")
    for name in ("retrieval_recall_at_k", "mrr", "answer_accuracy", "escalation_precision",
                 "escalation_recall", "task_completion"):
        print(f"  {name:24s} {metrics[name]:.0%}" if name != "mrr" else f"  {name:24s} {metrics[name]:.2f}")
    print(f"  {'latency p50 / p95':24s} {metrics['latency_p50_ms']:.0f} / {metrics['latency_p95_ms']:.0f} ms")
    failed = [r for r in rows if not r["correct"]]
    if failed:
        print("\nFailures to inspect:")
        for r in failed:
            print(f"  {r['id']}: {r['question']}  (escalated={r['escalated']}, top_score={r['top_score']})")
    return metrics


if __name__ == "__main__":
    run()
