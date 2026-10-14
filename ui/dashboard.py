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


st.set_page_config(
    page_title="XAI Combat Agent Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_rag_client(model_name: str = "llama3") -> ExplainableRAGClient:
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


def check_numerical_faithfulness(explanation: str, meta: dict, chosen_prob: float, alt_prob: float) -> dict:
    """
    Audits whether the generated LLM explanation faithfully cites the logged ground-truth
    figures (V(s), chosen probability, and alternative probability).
    """
    v_s_val = meta.get("value_estimate", 0.0)
    v_s_str = f"{v_s_val:.4f}"
    v_s_short = f"{v_s_val:.2f}"

    chosen_str_dec = f"{chosen_prob:.4f}"
    chosen_str_pct = f"{chosen_prob * 100:.1f}"

    alt_str_dec = f"{alt_prob:.4f}"
    alt_str_pct = f"{alt_prob * 100:.1f}"

    has_v_s = (v_s_str in explanation) or (v_s_short in explanation)
    has_chosen_p = (chosen_str_dec in explanation) or (chosen_str_pct in explanation)
    has_alt_p = (alt_str_dec in explanation) or (alt_str_pct in explanation)

    return {
        "cites_v_s": has_v_s,
        "cites_chosen_prob": has_chosen_p,
        "cites_alt_prob": has_alt_p,
        "all_verified": has_v_s and has_chosen_p and has_alt_p
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
        "**Interactive Policy Inspector & Counterfactual Explanation Engine** — "
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

    st.divider()

    # --- Bottom Section: Counterfactual RAG Query Interface ---
    st.subheader("💬 Counterfactual Explanation Engine (ChromaDB + Ollama Llama 3)")

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
        with st.spinner(f"Retrieving telemetry slice around Step {current_step} and querying local Llama 3 model..."):
            result = rag_client.answer_question(
                user_query=user_query,
                explicit_step=current_step,
                window=context_window
            )

        # Latency Metrics Banner
        lat_col1, lat_col2, lat_col3 = st.columns(3)
        lat_col1.metric("Vector DB Retrieval Latency", f"{result['retrieval_latency_ms']} ms", delta="Sub-second (<1000ms)")
        lat_col2.metric("LLM Generation Time", f"{result['llm_latency_ms'] / 1000:.2f} s")
        lat_col3.metric("Retrieved Context Frames", f"{len(result['retrieved_frames'])} frames")

        exp_col, raw_col = st.columns([1.35, 1.0])

        with exp_col:
            st.markdown("### 📋 Grounded Counterfactual Explanation")
            st.markdown(result["explanation"])

        with raw_col:
            st.markdown("### 🔬 Ground-Truth Verification Table")
            meta = {
                "value_estimate": current_frame.get("value_estimate", 0.0),
                "chosen_action_name": chosen_name
            }
            audit = check_numerical_faithfulness(result["explanation"], meta, chosen_prob, alt_prob)

            if audit["all_verified"]:
                st.success("✅ **Faithfulness Verified:** Explanation accurately cites logged $V(s)$, chosen $\\pi(a\vert{}s)$, and alternative $\\pi(a\vert{}s)$ figures.")
            else:
                st.info("ℹ️ **Ground-Truth Comparison:** Verify the generated explanation against the exact logged telemetry below.")

            verification_table = {
                "Telemetry Parameter": [
                    "Step Number",
                    "Executed Action",
                    "Executed Action Prob π(a|s)",
                    "Top Rejected Alternative",
                    "Rejected Alternative Prob π(a|s)",
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