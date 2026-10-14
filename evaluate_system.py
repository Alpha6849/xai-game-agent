import os
import json
import time
import statistics
from typing import List, Dict, Any

from rag.llm_client import ExplainableRAGClient
from rag.prompt_engine import CounterfactualPromptEngine


def evaluate_retrieval_performance(rag_client: ExplainableRAGClient, sample_steps: List[int]) -> Dict[str, Any]:
    """
    Benchmarks ChromaDB retrieval latency (ms) and exact target-frame recall accuracy
    across step-window queries and semantic natural-language queries.
    """
    latencies_ms = []
    exact_hits = 0

    for step in sample_steps:
        t0 = time.perf_counter()
        results = rag_client.retriever.retrieve_by_step(step=step, window=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved_steps = [r["metadata"].get("step") for r in results]
        if step in retrieved_steps:
            exact_hits += 1

    # Also test semantic vector similarity queries
    semantic_queries = [
        "agent retreated at low health",
        "agent chose shoot with clear line of sight",
        "agent moved to cover to block enemy fire",
        "negative critic value estimate disadvantageous state"
    ]
    semantic_latencies_ms = []
    for q in semantic_queries:
        t0 = time.perf_counter()
        _ = rag_client.retriever.retrieve_by_query(query=q, top_k=3)
        semantic_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    all_latencies = latencies_ms + semantic_latencies_ms

    return {
        "num_step_queries": len(sample_steps),
        "num_semantic_queries": len(semantic_queries),
        "step_retrieval_accuracy_pct": round((exact_hits / max(1, len(sample_steps))) * 100.0, 2),
        "mean_latency_ms": round(statistics.mean(all_latencies), 2),
        "median_latency_ms": round(statistics.median(all_latencies), 2),
        "min_latency_ms": round(min(all_latencies), 2),
        "max_latency_ms": round(max(all_latencies), 2),
        "sub_second_target_met": max(all_latencies) < 1000.0
    }


def evaluate_explanation_faithfulness(rag_client: ExplainableRAGClient, test_steps: List[int]) -> Dict[str, Any]:
    """
    Runs end-to-end counterfactual RAG queries on sample steps and audits whether
    the LLM output faithfully cites the exact logged V(s) and pi(a|s) figures.
    """
    records = []
    faithful_count = 0

    for step in test_steps:
        frames = rag_client.retriever.retrieve_by_step(step=step, window=1)
        if not frames:
            continue

        target_meta = next((f["metadata"] for f in frames if f["metadata"]["step"] == step), frames[0]["metadata"])
        chosen_name, chosen_prob, alt_name, alt_prob, _ = CounterfactualPromptEngine.extract_action_comparison(target_meta)
        v_s = float(target_meta.get("value_estimate", 0.0))

        query = f"Why did the agent choose '{chosen_name}' at step {step} instead of '{alt_name}'?"
        print(f"  -> Auditing LLM faithfulness for Step {step} ({chosen_name} vs {alt_name})...")

        res = rag_client.answer_question(user_query=query, explicit_step=step, window=1)
        explanation = res["explanation"]

        cites_vs = (f"{v_s:.4f}" in explanation) or (f"{v_s:.2f}" in explanation)
        cites_chosen = (f"{chosen_prob:.4f}" in explanation) or (f"{chosen_prob * 100:.1f}" in explanation)
        cites_alt = (f"{alt_prob:.4f}" in explanation) or (f"{alt_prob * 100:.1f}" in explanation)
        cites_actions = (chosen_name.lower() in explanation.lower()) and (alt_name.lower() in explanation.lower())

        is_faithful = cites_vs and cites_chosen and cites_alt and cites_actions
        if is_faithful:
            faithful_count += 1

        records.append({
            "step": step,
            "query": query,
            "chosen_action": chosen_name,
            "chosen_prob": chosen_prob,
            "alt_action": alt_name,
            "alt_prob": alt_prob,
            "v_s": v_s,
            "retrieval_latency_ms": res["retrieval_latency_ms"],
            "llm_latency_ms": res["llm_latency_ms"],
            "cites_v_s": cites_vs,
            "cites_chosen_prob": cites_chosen,
            "cites_alt_prob": cites_alt,
            "cites_action_names": cites_actions,
            "fully_faithful": is_faithful
        })

    total = max(1, len(records))
    return {
        "num_audited_explanations": len(records),
        "factual_faithfulness_rate_pct": round((faithful_count / total) * 100.0, 2),
        "samples": records
    }


def main():
    print("=" * 72)
    print("  XAI GAME AGENT — QUANTITATIVE EVALUATION SUITE (MODULE 4)")
    print("=" * 72)

    rag_client = ExplainableRAGClient(model_name="qwen3:4b")
    rag_client.retriever.ingest_game_rules("./data/game_rules.json")
    latest_session = rag_client.retriever.ingest_latest_session("./data/telemetry")
    print(f"[Init] Loaded & Indexed Session: {os.path.basename(latest_session)}")

    # 1. Benchmark ChromaDB Retrieval Latency & Accuracy across 25 step queries + 4 semantic queries
    sample_steps = list(range(5, 255, 10))
    print("\n[1/2] Benchmarking ChromaDB Vector Retrieval Latency & Accuracy...")
    retrieval_metrics = evaluate_retrieval_performance(rag_client, sample_steps)

    print(f"  - Step Queries Tested      : {retrieval_metrics['num_step_queries']}")
    print(f"  - Semantic Queries Tested  : {retrieval_metrics['num_semantic_queries']}")
    print(f"  - Retrieval Recall Accuracy: {retrieval_metrics['step_retrieval_accuracy_pct']}%")
    print(f"  - Mean Retrieval Latency   : {retrieval_metrics['mean_latency_ms']} ms")
    print(f"  - Median Retrieval Latency : {retrieval_metrics['median_latency_ms']} ms")
    print(f"  - Min / Max Latency        : {retrieval_metrics['min_latency_ms']} ms / {retrieval_metrics['max_latency_ms']} ms")
    print(f"  - Sub-Second Target (<1s)  : {'PASSED' if retrieval_metrics['sub_second_target_met'] else 'FAILED'}")

    # 2. Audit Counterfactual Explanation Faithfulness (2 representative steps to keep CPU runtime fast)
    print("\n[2/2] Auditing Counterfactual Explanation Faithfulness (Qwen3)...")
    faithfulness_metrics = evaluate_explanation_faithfulness(rag_client, test_steps=[10, 25])

    print(f"  - Explanations Audited     : {faithfulness_metrics['num_audited_explanations']}")
    print(f"  - Factual Faithfulness Rate: {faithfulness_metrics['factual_faithfulness_rate_pct']}%")

    report = {
        "session_evaluated": os.path.basename(latest_session),
        "retrieval_benchmark": retrieval_metrics,
        "faithfulness_audit": faithfulness_metrics
    }

    os.makedirs("./data", exist_ok=True)
    report_path = "./data/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"[Done] Saved full quantitative evaluation report to: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()