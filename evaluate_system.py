import glob
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Tuple

from rag.llm_client import ExplainableRAGClient
from rag.prompt_engine import CounterfactualPromptEngine


NUM_EXPLANATION_CASES = 5
DESCRIPTIVE_CASES = 2
MODEL_NAME = "qwen3:4b"


def _load_frames(session_path: str) -> List[Dict[str, Any]]:
    with open(session_path, "r", encoding="utf-8") as session_file:
        payload = json.load(session_file)
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        raise ValueError(f"Telemetry file has no frame list: {session_path}")
    return sorted(frames, key=lambda frame: int(frame.get("step", 0)))


def _sample_frames(frames: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Choose deterministic, spread-out frames without inventing step numbers."""
    if count <= 0 or not frames:
        return []
    actual_count = min(count, len(frames))
    indexes = [min(len(frames) - 1, int((i + 0.5) * len(frames) / actual_count))
               for i in range(actual_count)]
    return [frames[index] for index in indexes]


def _frame_action_comparison(frame: Dict[str, Any]) -> Tuple[str, float, str, float, Dict[str, float]]:
    probabilities = {
        str(name): float(probability)
        for name, probability in frame.get("action_probabilities", {}).items()
    }
    chosen_name = str(frame.get("chosen_action_name", "Unknown"))
    chosen_probability = probabilities.get(chosen_name, 0.0)
    alternatives = sorted(
        ((name, probability) for name, probability in probabilities.items()
         if name != chosen_name),
        key=lambda item: item[1],
        reverse=True,
    )
    alt_name, alt_probability = alternatives[0] if alternatives else ("None", 0.0)
    return chosen_name, chosen_probability, alt_name, alt_probability, probabilities


def evaluate_retrieval_performance(
    rag_client: ExplainableRAGClient,
    sample_steps: List[int],
    session_id: str,
) -> Dict[str, Any]:
    """Measure exact-step recall and retrieval latency within one recorded session."""
    latencies_ms = []
    exact_hits = 0

    for step in sample_steps:
        t0 = time.perf_counter()
        results = rag_client.retriever.retrieve_by_step(
            step=step, window=2, session_id=session_id
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        retrieved_steps = [result["metadata"].get("step") for result in results]
        if step in retrieved_steps:
            exact_hits += 1

    semantic_queries = [
        "agent retreated at low health",
        "agent chose shoot with clear line of sight",
        "agent moved to cover to block enemy fire",
        "negative critic value estimate disadvantageous state",
    ]
    semantic_latencies_ms = []
    for query in semantic_queries:
        t0 = time.perf_counter()
        rag_client.retriever.retrieve_by_query(
            query=query, top_k=3, where_filter={"session_id": session_id}
        )
        semantic_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    all_latencies = latencies_ms + semantic_latencies_ms
    return {
        "num_step_queries": len(sample_steps),
        "num_semantic_queries": len(semantic_queries),
        "step_retrieval_accuracy_pct": round(
            (exact_hits / max(1, len(sample_steps))) * 100.0, 2
        ),
        "mean_latency_ms": round(statistics.mean(all_latencies), 2) if all_latencies else None,
        "median_latency_ms": round(statistics.median(all_latencies), 2) if all_latencies else None,
        "min_latency_ms": round(min(all_latencies), 2) if all_latencies else None,
        "max_latency_ms": round(max(all_latencies), 2) if all_latencies else None,
        "sub_second_target_met": bool(all_latencies) and max(all_latencies) < 1000.0,
    }


def _build_question(test_type: str, step: int, chosen_name: str, alt_name: str) -> str:
    if test_type == "descriptive":
        if step % 2:
            return (
                f"Describe the recorded decision at step {step}, including the selected action, "
                "its strongest alternative, V(s), reward, health, ammo, and tactical flags."
            )
        return (
            f"What does the telemetry record at step {step} about the selected action, "
            "its strongest alternative, the critic value, and the agent's state?"
        )
    return (
        f"Why did the agent choose '{chosen_name}' at step {step} instead of '{alt_name}'?"
    )


def evaluate_explanation_faithfulness(
    rag_client: ExplainableRAGClient,
    session_path: str,
    num_cases: int = NUM_EXPLANATION_CASES,
    descriptive_cases: int = DESCRIPTIVE_CASES,
) -> Dict[str, Any]:
    """Run and preserve real answers, with each answer paired to its source frame."""
    frames = _load_frames(session_path)
    selected_frames = _sample_frames(frames, num_cases)
    session_id = os.path.splitext(os.path.basename(session_path))[0]
    records: List[Dict[str, Any]] = []

    for index, frame in enumerate(selected_frames):
        step = int(frame.get("step", 0))
        chosen_name, chosen_prob, alt_name, alt_prob, all_probs = _frame_action_comparison(frame)
        test_type = "descriptive" if index < descriptive_cases else "counterfactual"
        query = _build_question(test_type, step, chosen_name, alt_name)
        print(f"  -> Running {test_type} explanation {index + 1}/{len(selected_frames)} at step {step}...")

        result = rag_client.answer_question(
            user_query=query,
            explicit_step=step,
            window=0,
            session_id=session_id,
        )
        explanation = str(result.get("explanation", ""))
        raw_qwen_response = str(result.get("raw_model_response", ""))
        try:
            raw_payload = json.loads(raw_qwen_response)
            raw_takeaway = str(raw_payload.get("takeaway", "")) if isinstance(raw_payload, dict) else ""
        except (json.JSONDecodeError, TypeError):
            raw_takeaway = raw_qwen_response
        causal_phrase_pattern = re.compile(
            r"\b(?:because|due\s+to|based\s+on|therefore|thus|as\s+a\s+result|"
            r"in\s+order\s+to|to\s+avoid|to\s+maintain|to\s+ensure)\b",
            re.IGNORECASE,
        )
        raw_causal_phrase_flag = bool(causal_phrase_pattern.search(raw_takeaway))
        final_causal_phrase_flag = bool(causal_phrase_pattern.search(explanation))
        fallback_reason = result.get("fallback_reason")
        answer_source = (
            "error" if str(fallback_reason or "").startswith("Ollama inference failed")
            else "fallback" if result.get("explanation_fallback")
            else "qwen"
        )
        value_estimate = float(frame.get("value_estimate", 0.0))

        # The answer displays V(s) to four decimals and probabilities either as
        # the logged six-decimal value or as a one-decimal percentage.
        value_token = f"{value_estimate:.4f}"
        chosen_probability_tokens = (f"{chosen_prob:.6f}", f"{chosen_prob * 100.0:.1f}%")
        alt_probability_tokens = (f"{alt_prob:.6f}", f"{alt_prob * 100.0:.1f}%")
        value_match = value_token in explanation
        chosen_probability_match = any(token in explanation for token in chosen_probability_tokens)
        alt_probability_match = any(token in explanation for token in alt_probability_tokens)
        action_names_match = chosen_name.casefold() in explanation.casefold() and (
            alt_name == "None" or alt_name.casefold() in explanation.casefold()
        )

        retrieved_frames = result.get("retrieved_frames", [])
        validator_error = None
        if retrieved_frames:
            validator_error = rag_client._answer_validation_error(
                explanation, query, retrieved_frames, step
            )
        else:
            validator_error = "No telemetry frame was retrieved for the requested session and step."

        numeric_grounding_pass = (
            value_match and chosen_probability_match and alt_probability_match
            and action_names_match and validator_error is None
        )

        records.append({
            "test_id": f"T{index + 1:02d}",
            "question_type": test_type,
            "step": step,
            "question": query,
            "source_session": os.path.basename(session_path),
            "source_telemetry": {
                "chosen_action": chosen_name,
                "chosen_probability_logged": chosen_prob,
                "top_rejected_action": alt_name,
                "top_rejected_probability_logged": alt_prob,
                "all_action_probabilities_logged": all_probs,
                "value_estimate_logged": value_estimate,
                "reward_logged": frame.get("reward"),
                "observation_logged": frame.get("observation"),
                "flags_logged": frame.get("flags", {}),
            },
            "answer": explanation,
            "raw_qwen_response": raw_qwen_response,
            "raw_takeaway_causal_phrase_flag": raw_causal_phrase_flag,
            "final_answer_causal_phrase_flag": final_causal_phrase_flag,
            "answer_source": answer_source,
            "fallback_reason": fallback_reason,
            "retrieval_latency_ms": result.get("retrieval_latency_ms"),
            "llm_latency_ms": result.get("llm_latency_ms"),
            "checks": {
                "V_s_matches_logged_value_at_4dp": value_match,
                "chosen_probability_matches_logged_value": chosen_probability_match,
                "alternative_probability_matches_logged_value": alt_probability_match,
                "action_names_match_telemetry": action_names_match,
                "client_grounding_validator_error": validator_error,
                "automated_numeric_grounding_pass": numeric_grounding_pass,
                "manual_unsupported_claim_review": "pending",
            },
        })

    attempted_count = len(records)
    completed_records = [record for record in records if record["answer_source"] != "error"]
    completed_count = len(completed_records)
    attempted_type_counts = {
        "descriptive": sum(record["question_type"] == "descriptive" for record in records),
        "counterfactual": sum(record["question_type"] == "counterfactual" for record in records),
    }
    completed_type_counts = {
        "descriptive": sum(record["question_type"] == "descriptive" for record in completed_records),
        "counterfactual": sum(record["question_type"] == "counterfactual" for record in completed_records),
    }
    value_matches = sum(
        record["checks"]["V_s_matches_logged_value_at_4dp"] for record in completed_records
    )
    probability_matches = sum(
        record["checks"]["chosen_probability_matches_logged_value"]
        and record["checks"]["alternative_probability_matches_logged_value"]
        for record in completed_records
    )
    grounding_passes = sum(
        record["checks"]["automated_numeric_grounding_pass"] for record in completed_records
    )
    answer_source_counts = {
        source: sum(record["answer_source"] == source for record in records)
        for source in ("qwen", "fallback", "error")
    }
    raw_causal_flags = sum(record["raw_takeaway_causal_phrase_flag"] for record in records)
    final_causal_flags = sum(record["final_answer_causal_phrase_flag"] for record in completed_records)

    return {
        "num_queries_attempted": attempted_count,
        "num_explanations_sampled": completed_count,
        "failed_model_requests": attempted_count - completed_count,
        "question_type_counts_attempted": attempted_type_counts,
        "question_type_counts_answered": completed_type_counts,
        "answer_source_counts": answer_source_counts,
        "quoted_value_matches_at_4dp": f"{value_matches}/{completed_count}",
        "quoted_policy_probability_matches_at_logged_or_display_precision": f"{probability_matches}/{completed_count}",
        "automated_numeric_grounding_passes": f"{grounding_passes}/{completed_count}",
        "automated_numeric_grounding_rate_pct": round(
            (grounding_passes / completed_count) * 100.0, 2
        ) if completed_count else None,
        "raw_qwen_takeaways_with_causal_phrase_flags": f"{raw_causal_flags}/{attempted_count}",
        "final_answers_with_causal_phrase_flags": f"{final_causal_flags}/{completed_count}",
        "unsupported_claims_manual_review": {
            "status": "pending",
            "reason": (
                "Telemetry and citation checks cannot independently judge every natural-language claim; "
                "review the preserved answers before reporting an unsupported-claims count."
            ),
        },
        "method": {
            "sampling": "Five deterministic frames spread across the newest session; fewer if the session has fewer frames.",
            "question_mix": "First two descriptive, remaining three counterfactual (when five frames are available).",
            "probability_precision": "Accepts the logged six-decimal probability or the answer's one-decimal percentage display.",
            "score_limitation": "Automated numeric grounding is not a complete semantic faithfulness score.",
            "causal_phrase_screen": "A keyword screen for common causal phrases; it is a warning signal, not a complete unsupported-claim detector.",
        },
        "samples": records,
    }


def main() -> None:
    print("=" * 72)
    print("  XAI GAME AGENT — FIVE-QUERY TELEMETRY-GROUNDED EVALUATION")
    print("=" * 72)

    telemetry_files = glob.glob("./data/telemetry/session_*.json")
    if not telemetry_files:
        raise FileNotFoundError("No telemetry session files found in ./data/telemetry")
    latest_session = max(telemetry_files, key=os.path.getmtime)
    session_id = os.path.splitext(os.path.basename(latest_session))[0]
    frames = _load_frames(latest_session)
    if not frames:
        raise ValueError(f"No telemetry frames found in {latest_session}")

    rag_client = ExplainableRAGClient(model_name=MODEL_NAME)
    rag_client.retriever.ingest_game_rules("./data/game_rules.json")
    indexed_count = rag_client.retriever.ingest_telemetry_session(latest_session)
    print(f"[Init] Indexed {indexed_count} frames from {os.path.basename(latest_session)}")

    retrieval_frames = _sample_frames(frames, min(25, len(frames)))
    retrieval_steps = [int(frame.get("step", 0)) for frame in retrieval_frames]
    retrieval_metrics = evaluate_retrieval_performance(
        rag_client, retrieval_steps, session_id=session_id
    )
    print(
        f"[Retrieval] exact-step recall={retrieval_metrics['step_retrieval_accuracy_pct']}%; "
        f"mean latency={retrieval_metrics['mean_latency_ms']} ms"
    )

    faithfulness_metrics = evaluate_explanation_faithfulness(
        rag_client, latest_session, num_cases=NUM_EXPLANATION_CASES,
        descriptive_cases=DESCRIPTIVE_CASES,
    )
    print(
        "[Explanation audit] sampled="
        f"{faithfulness_metrics['num_explanations_sampled']}; "
        f"question types answered={faithfulness_metrics['question_type_counts_answered']}; "
        f"failed model requests={faithfulness_metrics['failed_model_requests']}; "
        "automated numeric grounding="
        f"{faithfulness_metrics['automated_numeric_grounding_passes']}"
    )
    print("[Manual review] Unsupported-claim count remains pending human review.")

    report = {
        "model": MODEL_NAME,
        "session_evaluated": os.path.basename(latest_session),
        "indexed_frames": indexed_count,
        "retrieval_benchmark": retrieval_metrics,
        "five_query_explanation_audit": faithfulness_metrics,
    }

    report_path = "./data/evaluation_report_5query.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    print(f"[Done] Full answers and paired telemetry saved to: {report_path}")


if __name__ == "__main__":
    main()
