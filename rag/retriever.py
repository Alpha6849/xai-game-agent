import os
import json
import glob
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class TelemetryRetriever:
    """
    Manages ChromaDB collections for runtime telemetry frames and static game rules.
    Supports exact step retrieval, nearest timestamp lookup, windowed context, and semantic search.
    """

    def __init__(self, persist_dir: str = "./data/chromadb"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.telemetry_collection = self.client.get_or_create_collection(
            name="agent_telemetry",
            metadata={"hnsw:space": "cosine"}
        )

        self.rules_collection = self.client.get_or_create_collection(
            name="game_rules",
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def serialize_frame_to_text(frame: Dict[str, Any]) -> str:
        """
        Converts a numerical telemetry frame into a structured natural-language summary
        suitable for text embedding while preserving key behavioral indicators.
        """
        step = frame.get("step", 0)
        timestamp = frame.get("timestamp", 0.0)
        chosen_name = frame.get("chosen_action_name", f"Action_{frame.get('chosen_action', 0)}")
        v_s = frame.get("value_estimate", 0.0)
        reward = frame.get("reward", 0.0)

        obs = frame.get("observation", [0.0] * 16)
        agent_hp = round(obs[5] * 100, 1) if len(obs) > 5 else 100.0
        opp_hp = round(obs[6] * 100, 1) if len(obs) > 6 else 100.0
        ammo = round(obs[7] * 30) if len(obs) > 7 else 30
        opp_dist = round(obs[4] * 600, 1) if len(obs) > 4 else 0.0

        flags = frame.get("flags", {})
        low_hp_str = "LOW HEALTH (<40%)" if flags.get("is_low_health") else "healthy"
        los_str = "HAS line-of-sight" if flags.get("has_los") else "NO line-of-sight"
        cover_str = "IN COVER" if flags.get("in_cover") else "exposed (not in cover)"

        probs: Dict[str, float] = frame.get("action_probabilities", {})
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)

        chosen_prob = probs.get(chosen_name, 0.0)
        alt_name, alt_prob = ("None", 0.0)
        for act_name, p in sorted_probs:
            if act_name != chosen_name:
                alt_name, alt_prob = act_name, p
                break

        summary = (
            f"Step {step} (Time {timestamp:.2f}s): Agent chose '{chosen_name}' (prob={chosen_prob:.3f}) "
            f"over highest alternative '{alt_name}' (prob={alt_prob:.3f}). "
            f"Critic state value V(s)={v_s:.4f}, step reward={reward:.3f}. "
            f"Status: Agent HP={agent_hp}%, Opponent HP={opp_hp}%, Ammo={ammo}/30, "
            f"Opponent Distance={opp_dist}px. Tactical state: {low_hp_str}, {los_str}, {cover_str}."
        )
        return summary

    def ingest_game_rules(self, rules_path: str = "./data/game_rules.json") -> int:
        """Indexes static game mechanics, action descriptions, and reward shaping weights."""
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Game rules file not found at: {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as f:
            rules_data = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for section_key, content in rules_data.items():
            doc_text = f"Game Rule Category [{section_key}]: {json.dumps(content)}"
            documents.append(doc_text)
            metadatas.append({"category": section_key})
            ids.append(f"rule_{section_key}")

        self.rules_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def ingest_telemetry_session(self, session_path: str, batch_size: int = 200) -> int:
        """
        Reads a recorded telemetry JSON file and indexes all frames into ChromaDB in batches.
        """
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Telemetry session not found: {session_path}")

        with open(session_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
        session_id = os.path.splitext(os.path.basename(session_path))[0]

        ids, documents, metadatas = [], [], []

        for frame in frames:
            step = int(frame.get("step", 0))
            doc_id = f"{session_id}_step_{step}"
            text_summary = self.serialize_frame_to_text(frame)

            obs = frame.get("observation", [0.0] * 16)
            flags = frame.get("flags", {})
            probs = frame.get("action_probabilities", {})

            meta = {
                "session_id": session_id,
                "step": step,
                "timestamp": float(frame.get("timestamp", 0.0)),
                "chosen_action": int(frame.get("chosen_action", 0)),
                "chosen_action_name": str(frame.get("chosen_action_name", "")),
                "value_estimate": float(frame.get("value_estimate", 0.0)),
                "reward": float(frame.get("reward", 0.0)),
                "done": bool(frame.get("done", False)),
                "is_low_health": bool(flags.get("is_low_health", False)),
                "has_los": bool(flags.get("has_los", False)),
                "in_cover": bool(flags.get("in_cover", False)),
                "agent_hp_pct": round(float(obs[5]) * 100, 1) if len(obs) > 5 else 100.0,
                "opp_hp_pct": round(float(obs[6]) * 100, 1) if len(obs) > 6 else 100.0,
                "ammo_count": int(round(float(obs[7]) * 30)) if len(obs) > 7 else 30,
                "action_probabilities_json": json.dumps(probs),
                "raw_frame_json": json.dumps({
                    k: v for k, v in frame.items() if k != "latent_vector"
                })
            }

            ids.append(doc_id)
            documents.append(text_summary)
            metadatas.append(meta)

            if len(ids) >= batch_size:
                self.telemetry_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                ids, documents, metadatas = [], [], []

        if ids:
            self.telemetry_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return len(frames)

    def ingest_latest_session(self, telemetry_dir: str = "./data/telemetry") -> str:
        """Finds and indexes the most recently recorded session JSON file."""
        pattern = os.path.join(telemetry_dir, "session_*.json")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No telemetry session files found in {telemetry_dir}")

        latest_file = max(files, key=os.path.getmtime)
        count = self.ingest_telemetry_session(latest_file)
        print(f"[Retriever] Indexed {count} frames from {os.path.basename(latest_file)}")
        return latest_file

    def retrieve_by_step(self, step: int, window: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves the exact target step alongside surrounding steps [step - window, step + window]
        for temporal context.
        """
        min_step = max(0, step - window)
        max_step = step + window

        results = self.telemetry_collection.get(
            where={
                "$and": [
                    {"step": {"$gte": min_step}},
                    {"step": {"$lte": max_step}}
                ]
            },
            include=["documents", "metadatas"]
        )

        combined = []
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            combined.append({"document": doc, "metadata": meta})

        combined.sort(key=lambda x: x["metadata"]["step"])
        return combined

    def retrieve_by_query(
        self,
        query: str,
        top_k: int = 3,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a cosine similarity vector search against logged telemetry frames.
        """
        query_kwargs = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.telemetry_collection.query(**query_kwargs)

        retrieved = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for doc, meta, dist in zip(docs, metas, dists):
                retrieved.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return retrieved

    def retrieve_all_rules(self) -> List[str]:
        """Returns all static game rules and reward shaping definitions."""
        res = self.rules_collection.get(include=["documents"])
        return res.get("documents", [])