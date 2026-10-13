# Explainable Autonomous Game Agents Using Deep Reinforcement Learning and Retrieval-Augmented Generation (RAG)

An explainable AI (XAI) framework pairing an autonomous 2D combat agent trained via Proximal Policy Optimization (PPO) with an in-situ, counterfactual Retrieval-Augmented Generation (RAG) engine.

---

## 📌 Overview

Standard Deep Reinforcement Learning (DRL) policies are black-box function approximators: they execute complex tactical policies but cannot articulate their internal decision logic. Dumping raw game telemetry into Large Language Models (LLMs) leads to context window saturation and hallucinations.

This project resolves this transparency barrier by:

1. Simulating a 2D adversarial combat arena wrapped in the Farama Gymnasium API.
2. Training an autonomous decision-making agent using PPO via Stable-Baselines3.
3. Logging runtime environment telemetry alongside internal actor-critic signals: value estimates $V(s)$, full action-probability distributions $\pi(a\vert{}s)$, and penultimate latent feature vectors.
4. Utilizing a vector store (ChromaDB) and local LLM (via Ollama) to answer natural-language user queries with grounded, counterfactual explanations (e.g., explaining why an agent retreated instead of engaging).

---

## 🏗️ System Architecture

The pipeline consists of three decoupled components:

1. **Simulation Engine (`env/`)**: Bounded 2D Pygame combat environment handling entity movement, line-of-sight raycasting, projectile resolution, and Gymnasium standardization.
2. **Policy Learning Engine (`agent/`)**: PyTorch/Stable-Baselines3 PPO implementation featuring discrete action branching and dense reward shaping (damage dealt, defensive positioning, stalling penalties).
3. **Explainability & XAI Engine (`telemetry/`, `rag/`, `ui/`)**:
   - **Telemetry Logger**: Captures structured state-action tuples and internal network activations at sample steps and trigger events.
   - **Vector Database**: ChromaDB collection storing serialized telemetry records and static game rule knowledge.
   - **RAG Pipeline**: Cosine similarity retriever assembling prompt contexts with raw logged metrics to drive counterfactual deduction in local LLMs.
   - **Dashboard**: Interactive user interface allowing evaluators to pause gameplay and inspect decisions.

---

## 📁 Repository Structure

```text
xai-game-agent/
├── env/
│   ├── arena.py            # Pygame mechanics, entities, obstacle collisions
│   └── gym_wrapper.py      # Gymnasium-compliant observation/action spaces
├── agent/
│   ├── callbacks.py        # Custom SB3 callback extracting V(s) and π(a|s)
│   ├── models.py           # Actor-Critic network definitions
│   ├── record_session.py   # Ground-truth evaluation session recorder
│   └── train.py            # PPO training pipeline and checkpointing
├── telemetry/
│   ├── logger.py           # In-process frame-level JSON telemetry serializer
│   └── schema.py           # Dataclass schemas for telemetry records
├── rag/
│   ├── llm_client.py       # Ollama integration (Llama 3 / Mistral)
│   ├── prompt_engine.py    # Structured counterfactual prompt synthesis
│   └── retriever.py        # ChromaDB indexing and vector search routines
├── ui/
│   └── dashboard.py        # Interactive Streamlit explanation console
├── data/
│   └── game_rules.json     # Static knowledge base of mechanics and rewards
├── verify_telemetry.py     # Structural audit script for logged telemetry
├── requirements.txt
└── README.md
```

---

## 📦 Installation & Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally with Llama 3 or Mistral:

```bash
ollama pull llama3:8b
```

### 2. Environment Setup

```bash
# Clone the repository
git clone [https://github.com/Alpha6849/xai-game-agent.git](https://github.com/Alpha6849/xai-game-agent.git)
cd xai-game-agent

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux/macOS:
# source venv/bin/activate

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
# Train the PPO agent (saves checkpoints to ./checkpoints/ and logs to ./tensorboard_logs/)
python -m agent.train --steps 10000 --checkpoint-freq 10000

# Run evaluation to generate the ground-truth actor-critic telemetry log
python -m agent.record_session --model ./checkpoints/ppo_combat_agent_final.zip --episodes 2

# Verify that internal signals (V(s), action distributions, latent vectors) were captured correctly
python verify_telemetry.py
```

---

## 👥 Contributors

- **Prathamesh Chalak** (`24BCT0286`)
- **Sourish Dutta** (`24BCT0282`)
