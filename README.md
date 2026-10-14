# Explainable Autonomous Game Agents Using Deep Reinforcement Learning and Retrieval-Augmented Generation (RAG)[cite: 1, 2]

An end-to-end Explainable Artificial Intelligence (XAI) framework that connects sub-symbolic policy optimization with symbolic natural-language reasoning[cite: 2]. The system trains an autonomous combat agent in a custom 2D arena using Proximal Policy Optimization (PPO), intercepts intermediate Actor-Critic representations, and utilizes a Retrieval-Augmented Generation (RAG) pipeline over ChromaDB to provide grounded, hallucination-resistant counterfactual explanations via a local Large Language Model[cite: 1, 2, 4, 8].

---

## System Overview & Problem Statement

Standard Deep Reinforcement Learning (DRL) policies function as uninterpretable black boxes[cite: 2]. Feeding raw numerical telemetry into general-purpose LLMs overflows context windows and causes hallucinations[cite: 1, 3].

This framework resolves the interpretability challenge through three architectural mechanisms:
* **Decoupled Simulation & Telemetry Logging:** The agent interacts synchronously with the Pygame/Gymnasium environment while runtime telemetry is captured via non-blocking callbacks[cite: 1, 2, 7].
* **Actor-Critic Signal Extraction:** The logger captures intermediate internal network activations alongside environment state: the critic's state-value estimate $V(s)$, the complete policy action-probability distribution $\pi(a\vert{}s)$, and the penultimate latent feature vector (128-d)[cite: 3, 4, 7].
* **Counterfactual RAG Grounding:** A ChromaDB vector database stores serialized frame summaries and static mechanics rules[cite: 1, 4, 8]. The prompt engine dynamically extracts the highest-probability rejected action ($\arg\max_{a \neq a_{\text{chosen}}} \pi(a\vert{}s)$) to force the LLM to contrast chosen vs. unchosen actions using logged numerical evidence[cite: 4, 8, 9].

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
|           |  - Local Ollama Inference (Llama 3 / Mistral)        |                 |
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

## Repository Structure

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

## Technical Specifications

### Observation Space (`spaces.Box(16,)`)
* `obs[0:2]`: Agent position coordinates $(x, y)$ normalized to $[0, 1]$[cite: 5].
* `obs[2:4]`: Relative opponent displacement $(\Delta x, \Delta y)$ normalized to $[-1, 1]$[cite: 5].
* `obs[4]`: Euclidean distance to opponent normalized to $[0, 1]$[cite: 5].
* `obs[5]`: Agent health ratio $[0, 1]$ with low-health threshold at $\le 0.40$[cite: 5, 7].
* `obs[6]`: Opponent health ratio $[0, 1]$[cite: 5].
* `obs[7]`: Ammunition ratio $[0, 1]$ relative to 30-round maximum[cite: 5].
* `obs[8]`: Weapon firing cooldown ratio $[0, 1]$[cite: 5].
* `obs[9]`: Line-of-sight indicator flag (`1.0` if unobstructed, `0.0` if blocked)[cite: 5].
* `obs[10:12]`: Relative vector to nearest defensive cover barrier $(\Delta x, \Delta y)$[cite: 5].
* `obs[12]`: Defensive cover flag (`1.0` if shielded behind cover, `0.0` if exposed)[cite: 5, 7].
* `obs[13:15]`: Relative trajectory vector of nearest incoming threat projectile[cite: 5].
* `obs[15]`: Incoming projectile threat active flag (`1.0` if threat present, `0.0` otherwise)[cite: 5].

### Action Space (`spaces.Discrete(7)`)
* `0`: Move North (Translates position upward)[cite: 7].
* `1`: Move South (Translates position downward)[cite: 7].
* `2`: Move East (Translates position rightward)[cite: 7].
* `3`: Move West (Translates position leftward)[cite: 7].
* `4`: Shoot (Discharges projectile along LOS; consumes 1 ammo)[cite: 7].
* `5`: Move to Cover (Navigates directly toward nearest barrier)[cite: 7].
* `6`: Retreat (Translates directly away from opponent)[cite: 7].

### Reward Structure
$$R_{\text{total}} = R_{\text{terminal}} + \sum R_{\text{shaping}}$$

| Component | Trigger Condition | Weight |
| :--- | :--- | :--- |
| **Terminal Win** | Opponent eliminated | `+10.0`[cite: 7] |
| **Terminal Loss** | Agent eliminated | `-10.0`[cite: 7] |
| **Hit Landed** | Projectile connects with opponent | `+0.5`[cite: 7] |
| **Damage Incurred** | Agent struck by enemy projectile | `-0.2`[cite: 7] |
| **Defensive Cover** | Holding cover position when health $< 40\%$ | `+0.3`[cite: 7] |
| **Stalling Penalty** | Per-step penalty applied to prevent idling | `-0.01`[cite: 7] |

---

## Installation & Setup

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

### 3. Language Model Initialization
Start the Ollama daemon and pull the target LLM:
```bash
ollama pull llama3
```
*(Alternative for lightweight CPU testing: `ollama pull qwen2.5:1.5b` or `ollama pull mistral`)*[cite: 5, 8].

---

## Execution Workflow

### Step 1: Policy Training
Train the agent using PPO with checkpointing and TensorBoard logging[cite: 1, 7]:
```bash
python -m agent.train
```

### Step 2: Evaluation Rollout & Telemetry Recording
Generate evaluation gameplay sessions and record Actor-Critic internal states[cite: 4, 7]:
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
Verify ChromaDB indexing, sub-second retrieval latency, and Llama 3 generation[cite: 3, 8]:
```bash
python test_rag_pipeline.py
```

### Step 5: Launch Explanation Dashboard
Start the Streamlit interface for interactive step inspection and counterfactual queries[cite: 8]:
```bash
streamlit run ui/dashboard.py
```
Access the dashboard at `http://localhost:8501`.

### Step 6: Run Quantitative Evaluation Suite
Benchmark retrieval latency, top-$k$ recall, and factual faithfulness[cite: 3, 9]:
```bash
python evaluate_system.py
```
*Output: Results exported to `data/evaluation_report.json`.*

---

## Experimental Benchmarks

Empirical performance measured across evaluation sessions (`session_1791910861.json`, 671 logged frames):

| Evaluation Metric | Target Benchmark | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Vector DB Retrieval Latency** | $< 1000\text{ ms}$ (Sub-second)[cite: 3, 9] | **1.93 ms (Median) / 31.94 ms (Mean)** | Passed |
| **Temporal Frame Recall Accuracy** | $100\%$ | **100.0%** | Passed |
| **Factual Faithfulness Rate** | Zero Hallucination[cite: 1, 9] | **100.0%** ($V(s)$ and $\pi(a\vert s)$ match logs)[cite: 9] | Passed |
| **Actor-Critic Signal Extraction** | $V(s)$, $\pi(a\vert s)$, Latent (128-d)[cite: 4, 7] | Verified across all frames | Passed |

---

## Project Authors & Academic Citation

### Project Team
* **Prathamesh Chalak** (Registration No: 24BCT0286) — School of Computer Science and Engineering, Vellore Institute of Technology, Vellore[cite: 1, 2]
* **Sourish Dutta** (Registration No: 24BCT0282) — School of Computer Science and Engineering, Vellore Institute of Technology, Vellore[cite: 1, 2]

This project was carried out as part of undergraduate academic research at Vellore Institute of Technology (VIT), Vellore[cite: 2].

```bibtex
@article{chalak_dutta_2026_xai,
  title={Explainable Autonomous Game Agents Using Deep Reinforcement Learning and Retrieval-Augmented Generation (RAG)},
  author={Chalak, Prathamesh and Dutta, Sourish},
  institution={School of Computer Science and Engineering, Vellore Institute of Technology},
  year={2026}
}
```
