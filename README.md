# 🛡️ Explainable Autonomous Game Agents Using DRL and RAG

An end-to-end Explainable Artificial Intelligence (XAI) framework that connects sub-symbolic policy optimization with symbolic natural-language reasoning. The system trains an autonomous combat agent in a custom 2D arena using Proximal Policy Optimization (PPO), intercepts intermediate Actor-Critic representations, and utilizes a Retrieval-Augmented Generation (RAG) pipeline over ChromaDB to provide grounded, hallucination-resistant counterfactual explanations via a local Large Language Model.

---

## 📌 System Overview & Problem Statement

Standard Deep Reinforcement Learning (DRL) policies function as uninterpretable black boxes. Feeding raw numerical telemetry into general-purpose LLMs overflows context windows and causes hallucinations.

This framework resolves the interpretability challenge through three architectural mechanisms:
* **Decoupled Simulation & Telemetry Logging:** The agent interacts synchronously with the Pygame/Gymnasium environment while runtime telemetry is captured via non-blocking callbacks.
* **Actor-Critic Signal Extraction:** The logger captures intermediate internal network activations alongside environment state: the critic's state-value estimate $V(s)$, the complete policy action-probability distribution $\pi(a\vert{}s)$, and the penultimate latent feature vector (128-d).
* **Counterfactual RAG Grounding:** A ChromaDB vector database stores serialized frame summaries and static mechanics rules. The prompt engine dynamically extracts the highest-probability rejected action (argmax over rejected actions π(a|s)) to force the LLM to contrast chosen vs. unchosen actions using logged numerical evidence.

---

## 🏗️ System Architecture

```
+------------------------------------------------------------------------------------+
|                                SYSTEM ARCHITECTURE                                 |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  +--------------------+        Observation        +-----------------------------+  |
|  |  Module 1: Arena   | ------------------------> |   Module 2: DRL Engine      |  |
|  |  2D Pygame Sim     | <------------------------ |   PPO Actor-Critic (SB3)    |  |
|  |  Gymnasium API     |       Sampled Action      |   - Policy Net: pi(a|s)     |  |
|  +--------------------+                           |   - Critic Net: V(s)        |  |
|           |                                       |   - Latent Vector (128-d)   |  |
|           |                                       +-----------------------------+  |
|           | State Snapshot                       Internal Tensors|                 |
|           +-----------------------+------------------------------+                 |
|                                   |                                                |
|                                   v                                                |
|           +------------------------------------------------------+                 |
|           |        Module 3: Telemetry & Vector Store            |                 |
|           |  - JSON Serialization / Sentence Embeddings          |                 |
|           |  - ChromaDB Persistent Store (Cosine Metric)         |                 |
|           |  - Static Game Rules Knowledge Base                  |                 |
|           +------------------------------------------------------+                 |
|                                   |                                                |
|                                   | Retrieved Context Window                       |
|                                   v                                                |
|           +------------------------------------------------------+                 |
|           |        Counterfactual RAG Engine                     |                 |
|           |  - Dynamic Action Contrast: Chosen vs. Top Rejected  |                 |
|           |  - Local Ollama Inference (Qwen3 / Mistral)        |                 |
|           +------------------------------------------------------+                 |
|                                   |                                                |
|                                   v Grounded Explanation & Latency                 |
|           +------------------------------------------------------+                 |
|           |   Module 4: Explanation Dashboard (Streamlit)        |                 |
|           |  - Interactive Step & Trigger Event Filtering        |                 |
|           |  - Real-Time Action Distribution Visualizer (Plotly) |                 |
|           |  - Automated Numerical Faithfulness Auditing         |                 |
|           +------------------------------------------------------+                 |
+------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
xai-game-agent/
├── agent/
│   ├── __init__.py
│   ├── callbacks.py          # TelemetryRecorderCallback extracting V(s), pi(a|s), and latents
│   ├── models.py             # Actor-Critic network configuration with 128-unit layers
│   ├── record_session.py     # Rollout script recording evaluation sessions to JSON
│   └── train.py              # PPO training pipeline with TensorBoard logging
├── data/
│   ├── chromadb/             # Persistent ChromaDB vector database files
│   ├── game_rules.json       # Domain knowledge base (actions, mechanics, reward weights)
│   └── telemetry/            # Runtime JSON session telemetry logs
├── env/
│   ├── __init__.py
│   ├── arena.py              # 2D Pygame simulation, raycasting LOS, combat logic
│   └── gym_wrapper.py        # Farama Gymnasium wrapper (observation & action spaces)
├── rag/
│   ├── __init__.py
│   ├── llm_client.py         # Local Ollama client with deterministic decoding
│   ├── prompt_engine.py      # Counterfactual prompt template enforcing 4-part structure
│   └── retriever.py          # Frame serializer, ChromaDB indexing, and hybrid search
├── telemetry/
│   ├── __init__.py
│   ├── logger.py             # Buffered session logger writing standardized JSON records
│   └── schema.py             # TelemetryFrame schema definition and data validation
├── ui/
│   ├── __init__.py
│   └── dashboard.py          # Streamlit evaluation dashboard and HUD inspector
├── documentation_and_abstract/
│   ├── Abstract.pdf
│   └── Entire_paper.pdf
├── evaluate_system.py        # Automated benchmarking suite (latency, recall, faithfulness)
├── test_rag_pipeline.py      # End-to-end integration test harness
├── verify_telemetry.py       # Corpus validation script auditing schema constraints
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Technical Specifications

###  Observation Space (`spaces.Box(16,)`)
* `obs[0:2]`: Agent position coordinates $(x, y)$ normalized to $[0, 1]$.
* `obs[2:4]`: Relative opponent displacement $(\Delta x, \Delta y)$ normalized to $[-1, 1]$.
* `obs[4]`: Euclidean distance to opponent normalized to $[0, 1]$.
* `obs[5]`: Agent health ratio $[0, 1]$ with low-health threshold at $\le 0.40$.
* `obs[6]`: Opponent health ratio $[0, 1]$.
* `obs[7]`: Ammunition ratio $[0, 1]$ relative to 30-round maximum.
* `obs[8]`: Weapon firing cooldown ratio $[0, 1]$.
* `obs[9]`: Line-of-sight indicator flag (`1.0` if unobstructed, `0.0` if blocked).
* `obs[10:12]`: Relative vector to nearest defensive cover barrier $(\Delta x, \Delta y)$.
* `obs[12]`: Defensive cover flag (`1.0` if shielded behind cover, `0.0` if exposed).
* `obs[13:15]`: Relative trajectory vector of nearest incoming threat projectile.
* `obs[15]`: Incoming projectile threat active flag (`1.0` if threat present, `0.0` otherwise).

###  Action Space (`spaces.Discrete(7)`)
* `0`: Move North (Translates position upward)
* `1`: Move South (Translates position downward)
* `2`: Move East (Rightward translation)
* `3`: Move West (Leftward translation)
* `4`: Shoot (Discharges projectile along LOS; consumes 1 ammo)
* `5`: Move to Cover (Navigates directly toward nearest barrier)
* `6`: Retreat (Translates directly away from opponent)

### ⚖️ Reward Shaping Function
$$R_{\text{total}} = R_{\text{terminal}} + \sum R_{\text{shaping}}$$

| Component | Trigger Condition | Weight |
| :--- | :--- | :--- |
| **Terminal Win** | Opponent eliminated | `+10.0` |
| **Terminal Loss** | Agent eliminated | `-10.0` |
| **Hit Landed** | Projectile connects with opponent | `+0.5` |
| **Damage Incurred** | Agent struck by enemy projectile | `-0.2` |
| **Defensive Cover** | Holding cover position when health $< 40\%$ | `+0.3` |
| **Stalling Penalty** | Per-step penalty applied to prevent idling | `-0.01` |

---

## 📦 Installation & Setup

### 1. Prerequisites
* Python 3.10 or 3.11
* [Ollama](https://ollama.com/) running locally

### 2. Environment Setup
```bash
git clone [https://github.com/Alpha6849/xai-game-agent.git](https://github.com/Alpha6849/xai-game-agent.git)
cd xai-game-agent

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Initialize Language Model
Start Ollama and pull the target LLM:
```bash
ollama pull qwen3:4b
```
*(Optional lightweight alternative for low-spec CPU setups: `ollama pull qwen2.5:1.5b` or `ollama pull mistral`)*

---

## 🕹️ Execution Workflow

### Step 1: Policy Training
Train the agent using PPO with checkpointing and TensorBoard logging:
```bash
python -m agent.train
```

### Step 2: Evaluation Rollout & Telemetry Recording
Generate evaluation gameplay sessions and record Actor-Critic internal states:
```bash
python -m agent.record_session
```
*Output: Serialized JSON logs saved to `data/telemetry/session_<id>.json`.*

### Step 3: Telemetry Schema Audit
Verify structural integrity, finite bounds, and probability sum invariants across all frames:
```bash
python verify_telemetry.py
```

### Step 4: RAG Pipeline Integration Test
Verify ChromaDB indexing, sub-second retrieval latency, and Qwen3 generation:
```bash
python test_rag_pipeline.py
```

### Step 5: Launch Explanation Dashboard
Start the Streamlit interface for interactive step inspection and counterfactual queries:
```bash
streamlit run ui/dashboard.py
```
Access the dashboard at `http://localhost:8501`.

### Step 6: Run Quantitative Evaluation Suite
Benchmark retrieval latency, top-k recall, and factual faithfulness:
```bash
python evaluate_system.py
```

---

## 👥 Authors & Academic Context

* **Prathamesh Chalak** (Registration No: 24BCT0286) — School of Computer Science and Engineering, Vellore Institute of Technology, Vellore
* **Sourish Dutta** (Registration No: 24BCT0282) — School of Computer Science and Engineering, Vellore Institute of Technology, Vellore
