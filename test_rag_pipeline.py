import json
from rag.llm_client import ExplainableRAGClient


def main():
    print("=== Initializing Explainable RAG Pipeline (Module 3) ===")
    rag_client = ExplainableRAGClient(model_name="llama3")

    # 1. Ingest Static Game Rules
    rules_count = rag_client.retriever.ingest_game_rules("./data/game_rules.json")
    print(f"[1/4] Indexed {rules_count} static game rule categories into ChromaDB.")

    # 2. Ingest Latest Telemetry Session
    latest_session = rag_client.retriever.ingest_latest_session("./data/telemetry")
    print(f"[2/4] Successfully ingested telemetry from: {latest_session}")

    # 3. Inspect a sample frame to formulate a concrete test query
    sample_frames = rag_client.retriever.retrieve_by_step(step=10, window=1)
    if not sample_frames:
        sample_frames = rag_client.retriever.retrieve_by_query("retreat low health cover shoot", top_k=1)

    target_meta = sample_frames[0]["metadata"]
    test_step = target_meta["step"]
    chosen_act = target_meta["chosen_action_name"]

    query = f"Why did the agent choose '{chosen_act}' at step {test_step} instead of an alternative action?"
    print(f"\n[3/4] Running Test Query: \"{query}\"")

    # 4. Run End-to-End Counterfactual RAG
    result = rag_client.answer_question(user_query=query, explicit_step=test_step, window=1)

    print("\n" + "=" * 70)
    print(f"RETRIEVAL LATENCY : {result['retrieval_latency_ms']} ms (Sub-second target < 1000 ms)")
    print(f"LLM INFERENCE TIME: {result['llm_latency_ms']} ms")
    print("=" * 70)

    print("\n--- GROUND-TRUTH LOGGED METRICS (RAW FIGURES) ---")
    print(f"Step              : {target_meta['step']}")
    print(f"Chosen Action     : {target_meta['chosen_action_name']}")
    print(f"Critic Value V(s) : {target_meta['value_estimate']:.4f}")
    print(f"Agent HP %        : {target_meta['agent_hp_pct']}% (Low Health: {target_meta['is_low_health']})")
    print(f"In Cover / LOS    : In-Cover={target_meta['in_cover']}, Has-LOS={target_meta['has_los']}")
    print(f"Probabilities     : {json.loads(target_meta['action_probabilities_json'])}")

    print("\n--- GENERATED COUNTERFACTUAL EXPLANATION (LLAMA 3) ---")
    print(result["explanation"])
    print("=" * 70)


if __name__ == "__main__":
    main()