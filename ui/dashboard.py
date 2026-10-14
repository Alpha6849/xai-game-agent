import os
import sys
import json
import glob
import re
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is in sys.path when launched via `streamlit run ui/dashboard.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.llm_client import ExplainableRAGClient
from rag.prompt_engine import CounterfactualPromptEngine
from env.gym_wrapper import CombatArenaEnv
from telemetry.schema import ACTION_NAMES


st.set_page_config(
    page_title="XAI Combat Agent Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_rag_client(model_name: str = "qwen3:4b") -> ExplainableRAGClient:
    """Initializes the RAG client and indexes static game rules once per session."""
    client = ExplainableRAGClient(
        model_name=model_name,
        persist_dir=os.path.join(PROJECT_ROOT, "data", "chromadb")
    )
    rules_path = os.path.join(PROJECT_ROOT, "data", "game_rules.json")
    if os.path.exists(rules_path):
        client.retriever.ingest_game_rules(rules_path)
    return client


@st.cache_data
def load_and_index_session(session_path: str):
    """Loads a telemetry JSON file from disk and ensures its frames are indexed in ChromaDB."""
    with open(session_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload

    client = init_rag_client()
    client.retriever.ingest_telemetry_session(session_path)
    return frames


def check_required_citations(explanation: str, meta: dict, chosen_prob: float, alt_probabilities) -> dict:
    """
    Checks whether selected logged values and labels appear in the generated text.
    This is a citation-presence check, not a validation of the explanation's reasoning.
    """
    def contains_number(value: str) -> bool:
        return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", explanation) is not None

    v_s_val = meta.get("value_estimate", 0.0)
    v_s_str = f"{v_s_val:.4f}"
    v_s_short = f"{v_s_val:.2f}"

    chosen_str_dec = f"{chosen_prob:.4f}"
    chosen_str_pct = f"{chosen_prob * 100:.1f}"

    has_v_s = contains_number(v_s_str) or contains_number(v_s_short)
    has_chosen_p = contains_number(chosen_str_dec) or contains_number(chosen_str_pct)
    if not isinstance(alt_probabilities, (list, tuple)):
        alt_probabilities = [alt_probabilities]
    has_alt_p = bool(alt_probabilities) and all(
        contains_number(f"{probability:.4f}") or contains_number(f"{probability * 100:.1f}")
        for probability in alt_probabilities
    )
    step = meta.get("step")
    has_step = step is not None and re.search(
        rf"\bstep\s+`?{re.escape(str(step))}\b", explanation, re.IGNORECASE
    ) is not None
    chosen_name = meta.get("chosen_action_name", "")
    has_chosen_action = bool(chosen_name) and chosen_name.casefold() in explanation.casefold()

    return {
        "cites_step": has_step,
        "cites_chosen_action": has_chosen_action,
        "cites_v_s": has_v_s,
        "cites_chosen_prob": has_chosen_p,
        "cites_alt_prob": has_alt_p,
        "all_required_figures_cited": all((has_step, has_chosen_action, has_v_s, has_chosen_p, has_alt_p))
    }


def main():
    rag_client = init_rag_client()

    # --- Sidebar: Session & Trigger Navigation ---
    st.sidebar.title("🎮 Telemetry Navigator")

    telemetry_dir = os.path.join(PROJECT_ROOT, "data", "telemetry")
    session_files = sorted(glob.glob(os.path.join(telemetry_dir, "session_*.json")), reverse=True)

    if not session_files:
        st.sidebar.error("No telemetry sessions found in `./data/telemetry/`. Run `python -m agent.record_session` first.")
        st.stop()

    selected_session_path = st.sidebar.selectbox(
        "Select Recorded Session",
        options=session_files,
        format_func=lambda p: os.path.basename(p)
    )

    frames = load_and_index_session(selected_session_path)
    total_frames = len(frames)

    st.sidebar.markdown(f"**Total Logged Frames:** `{total_frames}`")
    st.sidebar.divider()

    # Trigger Event Quick-Filter
    st.sidebar.subheader("⚡ Trigger Event Filter")
    filter_mode = st.sidebar.radio(
        "Jump to Specific Tactical Moments:",
        options=[
            "All Steps (Manual Slider)",
            "Retreat Decisions",
            "Shoot Actions",
            "Move to Cover Actions",
            "Low Health (<40% HP)",
            "In Defensive Cover"
        ]
    )

    filtered_indices = list(range(total_frames))
    if filter_mode == "Retreat Decisions":
        filtered_indices = [i for i, f in enumerate(frames) if f.get("chosen_action_name") == "Retreat"]
    elif filter_mode == "Shoot Actions":
        filtered_indices = [i for i, f in enumerate(frames) if f.get("chosen_action_name") == "Shoot"]
    elif filter_mode == "Move to Cover Actions":
        filtered_indices = [i for i, f in enumerate(frames) if f.get("chosen_action_name") == "Move to Cover"]
    elif filter_mode == "Low Health (<40% HP)":
        filtered_indices = [i for i, f in enumerate(frames) if f.get("flags", {}).get("is_low_health")]
    elif filter_mode == "In Defensive Cover":
        filtered_indices = [i for i, f in enumerate(frames) if f.get("flags", {}).get("in_cover")]

    if not filtered_indices:
        st.sidebar.warning(f"No frames matched '{filter_mode}' in this session. Showing all steps.")
        filtered_indices = list(range(total_frames))

    if filter_mode == "All Steps (Manual Slider)":
        selected_idx = st.sidebar.slider(
            "Select Simulation Step Index",
            min_value=0,
            max_value=max(0, total_frames - 1),
            value=min(9, max(0, total_frames - 1)),
            step=1
        )
    else:
        selected_idx = st.sidebar.selectbox(
            f"Matching Frames ({len(filtered_indices)} found)",
            options=filtered_indices,
            format_func=lambda idx: (
                f"Step {frames[idx].get('step', idx)} | "
                f"{frames[idx].get('chosen_action_name')} | "
                f"V(s)={frames[idx].get('value_estimate', 0.0):.3f}"
            )
        )

    current_frame = frames[selected_idx]
    current_step = int(current_frame.get("step", selected_idx))
    obs = current_frame.get("observation", [0.0] * 16)
    flags = current_frame.get("flags", {})
    probs = current_frame.get("action_probabilities", {})

    # Identify chosen action vs top rejected alternative
    chosen_name = current_frame.get("chosen_action_name", "Unknown")
    chosen_prob = float(probs.get(chosen_name, 0.0))
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    alt_name, alt_prob = "None", 0.0
    for act_name, p in sorted_probs:
        if act_name != chosen_name:
            alt_name, alt_prob = act_name, float(p)
            break

    # --- Main Header ---
    st.title("🛡️ Explainable Autonomous Game Agent (DRL + RAG)")
    st.markdown(
        "**Interactive Policy Inspector & Telemetry-Grounded Explanations** — "
        "Grounds natural-language explanations in PPO Actor-Critic signals $\\pi(a\vert{}s)$ and $V(s)$."
    )

    # --- Top Row: HUD Metrics & Action Probability Bar Chart ---
    col_hud, col_probs = st.columns([1.05, 1.25])

    with col_hud:
        st.subheader(f"📍 Arena Telemetry Snapshot — Step `{current_step}`")

        agent_hp = round(float(obs[5]) * 100, 1) if len(obs) > 5 else 100.0
        opp_hp = round(float(obs[6]) * 100, 1) if len(obs) > 6 else 100.0
        ammo = int(round(float(obs[7]) * 30)) if len(obs) > 7 else 30
        opp_dist = round(float(obs[4]) * 600, 1) if len(obs) > 4 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agent HP", f"{agent_hp}%")
        m2.metric("Opponent HP", f"{opp_hp}%")
        m3.metric("Ammo Pool", f"{ammo}/30")
        m4.metric("Enemy Dist", f"{opp_dist} px")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🧠 Actor-Critic Signals")
            st.write(f"**Executed Action:** `{chosen_name}` ({chosen_prob * 100:.1f}%)")
            st.write(f"**Top Alternative:** `{alt_name}` ({alt_prob * 100:.1f}%)")
            st.write(f"**Critic Value $V(s)$:** `{current_frame.get('value_estimate', 0.0):.4f}`")
            st.write(f"**Immediate Reward:** `{current_frame.get('reward', 0.0):.4f}`")

        with c2:
            st.markdown("##### 🚩 Tactical Environment Flags")
            st.write("**In Defensive Cover:**", "✅ Yes" if flags.get("in_cover") else "❌ No (Exposed)")
            st.write("**Line-of-Sight (LOS):**", "✅ Clear LOS" if flags.get("has_los") else "🧱 Blocked")
            st.write("**Low Health (<40%):**", "⚠️ Critical" if flags.get("is_low_health") else "🟢 Healthy")
            st.write("**Timestamp:**", f"`{current_frame.get('timestamp', 0.0):.2f}s`")

    with col_probs:
        st.subheader("🎯 Policy Action Distribution $\\pi(a\vert{}s)$")
        if probs:
            action_labels = list(probs.keys())
            prob_pcts = [float(probs[a]) * 100.0 for a in action_labels]

            bar_colors = []
            for a in action_labels:
                if a == chosen_name:
                    bar_colors.append("#2ecc71")  # Green for executed action
                elif a == alt_name:
                    bar_colors.append("#f39c12")  # Orange for top rejected alternative
                else:
                    bar_colors.append("#3498db")  # Blue for other candidates

            fig_probs = go.Figure(go.Bar(
                x=prob_pcts,
                y=action_labels,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{p:.2f}%" for p in prob_pcts],
                textposition="outside"
            ))
            fig_probs.update_layout(
                xaxis_title="Action Probability (%)",
                xaxis_range=[0, max(100, max(prob_pcts) + 15)],
                yaxis=dict(autorange="reversed"),
                height=330,
                margin=dict(l=10, r=30, t=10, b=20),
                legend_title_text="Green = Executed | Orange = Top Rejected Alternative"
            )
            st.plotly_chart(fig_probs, use_container_width=True)
            st.caption("🟢 **Green**: Executed Action | 🟠 **Orange**: Highest-Probability Rejected Alternative | 🔵 **Blue**: Other Actions")

    # --- Collapsible Episode Trajectory Chart ---
    with st.expander("📈 View Episode-Wide Critic Value V(s) & Health Trajectory", expanded=False):
        steps_list = [f.get("step", i) for i, f in enumerate(frames)]
        vs_list = [f.get("value_estimate", 0.0) for f in frames]
        hp_list = [round(f.get("observation", [0]*16)[5] * 100, 1) for f in frames]

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=steps_list, y=vs_list, mode="lines", name="Critic Value V(s)",
            line=dict(color="#9b59b6", width=2)
        ))
        fig_timeline.add_vline(
            x=current_step, line_dash="dash", line_color="#e74c3c",
            annotation_text=f"Step {current_step}"
        )
        fig_timeline.update_layout(
            xaxis_title="Simulation Step",
            yaxis_title="Critic State-Value Estimate V(s)",
            height=260,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with st.expander("🧪 Replay one alternative action from this exact state", expanded=False):
        pre_action_state = current_frame.get("pre_action_state")
        alternative_names = [name for name in probs if name != chosen_name]
        if not pre_action_state:
            st.info("This session has no saved pre-action snapshots. Record a new session to enable replay.")
        elif not alternative_names:
            st.info("This frame has no alternative action probabilities to replay.")
        else:
            alternative_name = st.selectbox(
                "Choose an unselected action to simulate for one step:",
                options=alternative_names,
                key="counterfactual_action"
            )
            st.caption(
                "The replay starts from the saved simulator state and random-number state. "
                "It compares one step only; it does not predict a full future episode."
            )
            if st.button("Run one-step replay", key="run_counterfactual_replay"):
                action_id = next(
                    (idx for idx, name in ACTION_NAMES.items() if name == alternative_name),
                    None
                )
                replay_env = CombatArenaEnv()
                try:
                    if action_id is None:
                        raise ValueError(f"Unknown action in telemetry: {alternative_name}")
                    replay_env.restore_state(pre_action_state)
                    simulated_obs, simulated_reward, terminated, truncated, replay_info = replay_env.step(action_id)

                    st.write(f"**Simulated action:** {alternative_name}")
                    actual_reward = float(current_frame.get("reward", 0.0))
                    st.metric(
                        "One-step reward",
                        f"{simulated_reward:.4f}",
                        delta=f"{simulated_reward - actual_reward:+.4f} vs logged action"
                    )
                    actual_obs = current_frame.get("next_observation")
                    if actual_obs:
                        actual_hp = float(actual_obs[5]) * 100
                        actual_opp_hp = float(actual_obs[6]) * 100
                        actual_ammo = float(actual_obs[7]) * 30
                        alt_hp = float(simulated_obs[5]) * 100
                        alt_opp_hp = float(simulated_obs[6]) * 100
                        alt_ammo = float(simulated_obs[7]) * 30
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Agent HP after step", f"{alt_hp:.1f}%", delta=f"{alt_hp - actual_hp:+.1f}% vs logged")
                        c2.metric("Opponent HP after step", f"{alt_opp_hp:.1f}%", delta=f"{alt_opp_hp - actual_opp_hp:+.1f}% vs logged")
                        c3.metric("Ammo after step", f"{alt_ammo:.0f}/30", delta=f"{alt_ammo - actual_ammo:+.0f} vs logged")
                    st.write("**Simulated step events:**")
                    st.json(replay_info.get("events", {}))
                    if terminated or truncated:
                        st.caption("The simulated step ended or reached the episode step limit.")
                except Exception as exc:
                    st.error(f"Could not replay this alternative action: {exc}")
                finally:
                    replay_env.close()

    st.divider()

    # --- Bottom Section: Counterfactual RAG Query Interface ---
    st.subheader("💬 Telemetry-Grounded Explanation (ChromaDB + Ollama Qwen3)")
    st.caption(
        "The text explanation uses recorded policy and telemetry. The replay panel above can simulate "
        "one alternative step, but it does not predict a full future episode."
    )

    preset_questions = [
        f"Why did the agent choose '{chosen_name}' at step {current_step} instead of '{alt_name}'?",
        f"Why did the agent not choose 'Shoot' or 'Retreat' at step {current_step}?",
        f"How did the critic value V(s) and tactical flags influence the decision at step {current_step}?"
    ]

    selected_preset = st.selectbox("Select a Question Template (or edit below):", options=preset_questions)
    user_query = st.text_input("Natural-Language Evaluator Query:", value=selected_preset)

    col_btn, col_window = st.columns([1, 3])
    with col_window:
        context_window = st.slider("Temporal Retrieval Window (± steps around target frame)", 0, 5, 2)

    if col_btn.button("🔍 Generate Grounded Explanation", type="primary", use_container_width=True):
        with st.spinner(f"Retrieving telemetry around Step {current_step} and querying local Qwen3 model..."):
            result = rag_client.answer_question(
                user_query=user_query,
                explicit_step=current_step,
                window=context_window,
                session_id=os.path.splitext(os.path.basename(selected_session_path))[0]
            )

        # Latency Metrics Banner
        lat_col1, lat_col2, lat_col3 = st.columns(3)
        lat_col1.metric("Vector DB Retrieval Latency", f"{result['retrieval_latency_ms']} ms", delta="Sub-second (<1000ms)")
        lat_col2.metric("LLM Generation Time", f"{result['llm_latency_ms'] / 1000:.2f} s")
        lat_col3.metric("Retrieved Context Frames", f"{len(result['retrieved_frames'])} frames")

        exp_col, raw_col = st.columns([1.35, 1.0])

        with exp_col:
            st.markdown("### 📋 Grounded Policy Explanation")
            if result.get("explanation_fallback"):
                st.caption("Qwen's takeaway did not pass a grounding check; showing the verified telemetry summary.")
                with st.expander("Qwen response diagnostics"):
                    st.write(result.get("fallback_reason", "No rejection detail was recorded."))
                    raw_response = result.get("raw_model_response", "")
                    st.code(raw_response if raw_response else "(Qwen returned no response text.)")
            st.markdown(result["explanation"].replace("\n", "  \n"))

        with raw_col:
            st.markdown("### 🔬 Logged Telemetry Comparison")
            meta = {
                "step": current_step,
                "value_estimate": current_frame.get("value_estimate", 0.0),
                "chosen_action_name": chosen_name
            }
            normalized_query = re.sub(r"[^a-z0-9]+", " ", user_query.casefold()).strip()
            requested_alt_actions = []
            for action_name, probability in probs.items():
                normalized_action = re.sub(r"[^a-z0-9]+", " ", action_name.casefold()).strip()
                if (action_name != chosen_name and normalized_action
                        and re.search(rf"(?<!\w){re.escape(normalized_action)}(?!\w)", normalized_query)):
                    requested_alt_actions.append((action_name, float(probability)))
            if not requested_alt_actions and alt_name != "None":
                requested_alt_actions = [(alt_name, alt_prob)]
            requested_alt_probabilities = [probability for _, probability in requested_alt_actions]
            audit = check_required_citations(
                result["explanation"], meta, chosen_prob, requested_alt_probabilities
            )

            if audit["all_required_figures_cited"]:
                st.info("The answer includes the checked step, action, and key logged figures. This citation check does not validate the reasoning or any untested alternative outcome.")
            else:
                st.warning("Some expected step, action, or figure citations are missing. Compare the answer with the logged telemetry below.")

            verification_table = {
                "Telemetry Parameter": [
                    "Step Number",
                    "Executed Action",
                    "Executed Action Prob π(a|s)",
                    "Top Rejected Alternative",
                    "Rejected Alternative Prob π(a|s)",
                    "Requested Alternative(s)",
                    "Critic Value Estimate V(s)",
                    "Immediate Step Reward",
                    "Agent Health (%)",
                    "Low Health Flag (<40%)",
                    "In Cover Flag",
                    "Line-of-Sight Flag"
                ],
                "Logged Ground-Truth Value": [
                    str(current_step),
                    chosen_name,
                    f"{chosen_prob:.4f} ({chosen_prob * 100:.1f}%)",
                    alt_name,
                    f"{alt_prob:.4f} ({alt_prob * 100:.1f}%)",
                    "; ".join(
                        f"{name}: {probability:.4f} ({probability * 100:.1f}%)"
                        for name, probability in requested_alt_actions
                    ) or "None identified",
                    f"{current_frame.get('value_estimate', 0.0):.4f}",
                    f"{current_frame.get('reward', 0.0):.4f}",
                    f"{agent_hp}%",
                    str(flags.get("is_low_health", False)),
                    str(flags.get("in_cover", False)),
                    str(flags.get("has_los", False))
                ]
            }
            st.table(verification_table)

        with st.expander("🛠️ Inspect Full Constructed RAG Prompt & Retrieved Documents"):
            st.code(result["constructed_prompt"], language="markdown")


if __name__ == "__main__":
    main()
