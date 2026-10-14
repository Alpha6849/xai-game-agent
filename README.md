# Explainable Autonomous Game Agents Using Deep Reinforcement Learning and Retrieval-Augmented Generation (RAG)

An explainable AI (XAI) framework pairing an autonomous 2D combat agent trained via Proximal Policy Optimization (PPO) with an in-situ, counterfactual Retrieval-Augmented Generation (RAG) engine.

---

## 📌 Project Status

- **Module 1 — 2D Simulation & Gymnasium Wrapper (`env/`)**: ✅ Implemented & Verified
- **Module 2 — PPO Policy & Actor-Critic Telemetry (`agent/`, `telemetry/`)**: ✅ Implemented & Verified
- **Module 3 — Vector Store & Counterfactual RAG (`rag/`)**: 🚧 In Progress (Scaffolding)
- **Module 4 — Interactive Explanation Dashboard (`ui/`)**: 📅 Planned (Scaffolding)

---

## 📌 Overview

Standard Deep Reinforcement Learning (DRL) policies are black-box function approximators: they execute complex tactical policies but cannot articulate their internal decision logic. Dumping raw game telemetry into Large Language Models (LLMs) leads to context window saturation and hallucinations.

This project resolves this transparency barrier by:

1. Simulating a 2D adversarial combat arena wrapped in the Farama Gymnasium API.
2. Training an autonomous decision-making agent using PPO via Stable-Baselines3.
3. Logging runtime environment telemetry alongside internal actor-critic signals: value estimates $V(s)$, full action-probability distributions $\pi(a\vert{}s)$, and penultimate latent feature vectors.
4. *(Planned — Module 3 & 4)* Utilizing a vector store (ChromaDB) and local LLM (via Ollama) to answer natural-language user queries with grounded, counterfactual explanations.

---

## 📁 Repository Structure

```text
xai-game-agent/
├── env/
│   ├── arena.py            # [Implemented] Pygame mechanics, entities, obstacle collisions
│   └── gym_wrapper.py      # [Implemented] Gymnasium-compliant 16-dim obs / 7-action spaces
├── agent/
│   ├── callbacks.py        # [Implemented] SB3 callback extracting V(s), π(a|s), latent vectors
│   ├── models.py           # [Implemented] Actor-Critic MLP network definitions
│   ├── record_session.py   # [Implemented] Ground-truth evaluation session recorder
│   └── train.py            # [Implemented] PPO training pipeline and checkpointing
├── telemetry/
│   ├── logger.py           # [Implemented] Frame-level JSON telemetry serializer
│   └── schema.py           # [Implemented] Dataclass schemas for telemetry records
├── rag/
│   ├── llm_client.py       # [Planned - Module 3] Ollama integration (Llama 3 / Mistral)
│   ├── prompt_engine.py    # [Planned - Module 3] Counterfactual prompt synthesis
│   └── retriever.py        # [Planned - Module 3] ChromaDB indexing and vector search
├── ui/
│   └── dashboard.py        # [Planned - Module 4] Interactive Streamlit explanation console
├── verify_telemetry.py     # [Implemented] Full-corpus audit script for logged telemetry
├── requirements.txt
└── README.md
```

---

## 📦 Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/Alpha6849/xai-game-agent.git](https://github.com/Alpha6849/xai-game-agent.git)
cd xai-game-agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Verification

### Step 1: Verify Gymnasium Arena (Module 1)

```bash
python -c "from env.gym_wrapper import CombatArenaEnv; from gymnasium.utils.env_checker import check_env; env = CombatArenaEnv(); check_env(env.unwrapped); print('Gymnasium Check Passed!')"
```

### Step 2: Train Agent & Extract Telemetry (Module 2)

```bash
# Train the PPO agent
python -m agent.train --steps 10000 --checkpoint-freq 10000

# Run evaluation to generate the ground-truth actor-critic telemetry log
python -m agent.record_session --model ./checkpoints/ppo_combat_agent_final.zip --episodes 2

# Audit all frames in the latest telemetry log for finite math, probability sums, and schema compliance
python verify_telemetry.py
```

---

## 👥 Contributors

- **Prathamesh Chalak** (`24BCT0286`)
- **Sourish Dutta** (`24BCT0282`)
