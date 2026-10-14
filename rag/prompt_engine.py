import json
from typing import List, Dict, Any, Tuple


class CounterfactualPromptEngine:
    """
    Constructs grounded counterfactual prompts that force the LLM to justify
    why the chosen action was preferred over the runner-up alternative using
    logged Actor-Critic signals V(s) and pi(a|s).
    """

    SYSTEM_PROMPT = (
        "You are an Explainable AI (XAI) tactical analyst for a Deep Reinforcement Learning "
        "combat agent trained with Proximal Policy Optimization (PPO). "
        "Your strict mandate is to explain the agent's behavior using ONLY the provided "
        "ground-truth telemetry figures and game rules. "
        "NEVER invent, guess, or hallucinate numbers not explicitly present in the prompt.\n\n"
        "For every explanation, you MUST follow this exact 4-part structure:\n"
        "1. **Action Executed**: State the exact step, action chosen, and its policy probability pi(a|s).\n"
        "2. **Counterfactual Alternative Rejected**: State the highest-probability alternative action "
        "that was NOT chosen (or the specific action the user asked about) and its exact probability pi(a|s).\n"
        "3. **Critic State Assessment V(s) & Environment Context**: Cite the critic's scalar value estimate V(s) "
        "and key operational flags (Health %, Ammo, Line-of-Sight, In-Cover status).\n"
        "4. **Grounded Tactical Reasoning**: Connect V(s), pi(a|s), and the game's reward shaping weights "
        "to explain why the policy network favored the chosen action over the rejected alternative."
    )

    @staticmethod
    def extract_action_comparison(meta: Dict[str, Any]) -> Tuple[str, float, str, float, Dict[str, float]]:
        """
        Identifies the chosen action probability and the highest-probability rejected alternative.
        """
        chosen_name = meta.get("chosen_action_name", "Unknown")
        probs_json = meta.get("action_probabilities_json", "{}")
        try:
            probs: Dict[str, float] = json.loads(probs_json)
        except json.JSONDecodeError:
            probs = {}

        chosen_prob = float(probs.get(chosen_name, 0.0))

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        alt_name, alt_prob = "None", 0.0
        for act, p in sorted_probs:
            if act != chosen_name:
                alt_name, alt_prob = act, float(p)
                break

        return chosen_name, chosen_prob, alt_name, alt_prob, probs

    def build_counterfactual_prompt(
        self,
        user_query: str,
        retrieved_frames: List[Dict[str, Any]],
        game_rules: List[str],
        focus_step: int = None
    ) -> str:
        """
        Builds a deterministic prompt combining static game rules, primary decision frame
        actor-critic signals, and surrounding frame context.
        """
        if not retrieved_frames:
            return (
                f"User Question: {user_query}\n"
                "Context: No matching telemetry frames were found in the database. "
                "State clearly that no telemetry data is available for this query."
            )

        primary_frame = retrieved_frames[0]
        if focus_step is not None:
            for item in retrieved_frames:
                if item["metadata"].get("step") == focus_step:
                    primary_frame = item
                    break

        meta = primary_frame["metadata"]
        chosen_name, chosen_prob, alt_name, alt_prob, all_probs = self.extract_action_comparison(meta)

        probs_formatted = ", ".join([f"{k}: {v:.4f}" for k, v in all_probs.items()])
        rules_block = "\n".join(game_rules)

        surrounding_context = []
        for item in retrieved_frames:
            step_num = item["metadata"].get("step")
            marker = "--> [TARGET DECISION FRAME]" if step_num == meta.get("step") else "    [CONTEXT FRAME]"
            surrounding_context.append(f"{marker} {item['document']}")
        timeline_block = "\n".join(surrounding_context)

        prompt = f"""=== STATIC GAME RULES & REWARD SHAPING WEIGHTS ===
{rules_block}

=== RETRIEVED TELEMETRY WINDOW ===
{timeline_block}

=== TARGET FRAME ACTOR-CRITIC SIGNALS (STEP {meta.get('step')}) ===
- Timestamp: {meta.get('timestamp', 0.0):.3f}s
- Executed Action: {chosen_name} (Probability pi(a|s) = {chosen_prob:.4f} / {chosen_prob*100:.1f}%)
- Top Rejected Alternative: {alt_name} (Probability pi(a|s) = {alt_prob:.4f} / {alt_prob*100:.1f}%)
- Full Action Probability Distribution pi(a|s): {{{probs_formatted}}}
- Critic State-Value Estimate V(s): {meta.get('value_estimate', 0.0):.4f}
- Immediate Step Reward: {meta.get('reward', 0.0):.4f}
- Agent Health: {meta.get('agent_hp_pct', 100.0)}% (Low Health Flag: {meta.get('is_low_health')})
- Opponent Health: {meta.get('opp_hp_pct', 100.0)}%
- Agent Ammo: {meta.get('ammo_count', 30)}/30
- Tactical Flags: Line-of-Sight={meta.get('has_los')}, In-Cover={meta.get('in_cover')}

=== USER QUESTION ===
"{user_query}"

=== INSTRUCTIONS ===
Answer the user's question by strictly comparing the Executed Action ({chosen_name}, pi={chosen_prob:.4f}) against the Rejected Alternative ({alt_name}, pi={alt_prob:.4f}, or any specific alternative mentioned in the question). Explicitly cite V(s) = {meta.get('value_estimate', 0.0):.4f} and the relevant environmental flags and reward weights above. Do not fabricate any numbers."""

        return prompt