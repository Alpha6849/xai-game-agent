import re
import time
from typing import Dict, Any, Optional
import ollama

from rag.retriever import TelemetryRetriever
from rag.prompt_engine import CounterfactualPromptEngine


class ExplainableRAGClient:
    """
    End-to-end RAG coordinator connecting ChromaDB telemetry retrieval,
    counterfactual prompt construction, and local Ollama LLM inference.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        persist_dir: str = "./data/chromadb",
        temperature: float = 0.1
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.retriever = TelemetryRetriever(persist_dir=persist_dir)
        self.prompt_engine = CounterfactualPromptEngine()

    @staticmethod
    def _extract_step_from_query(query: str) -> Optional[int]:
        """Detects if the user explicitly asked about a step number (e.g., 'step 42')."""
        match = re.search(r"\bstep\s+(\d+)\b", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def answer_question(
        self,
        user_query: str,
        explicit_step: Optional[int] = None,
        window: int = 2,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieves relevant telemetry frames, builds a counterfactual prompt,
        queries the local Ollama model, and returns the explanation + raw evidence.
        """
        t_start = time.perf_counter()

        target_step = explicit_step if explicit_step is not None else self._extract_step_from_query(user_query)

        if target_step is not None:
            retrieved_frames = self.retriever.retrieve_by_step(step=target_step, window=window)
        else:
            retrieved_frames = self.retriever.retrieve_by_query(query=user_query, top_k=top_k)

        game_rules = self.retriever.retrieve_all_rules()
        t_retrieval = time.perf_counter() - t_start

        prompt = self.prompt_engine.build_counterfactual_prompt(
            user_query=user_query,
            retrieved_frames=retrieved_frames,
            game_rules=game_rules,
            focus_step=target_step
        )

        t_llm_start = time.perf_counter()
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_engine.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "num_gpu": 0
                }
            )
            explanation_text = response["message"]["content"]
        except Exception as e:
            explanation_text = (
                f"[Ollama Inference Error]: Could not reach local model '{self.model_name}'. "
                f"Verify Ollama is running (`ollama serve`) and model is pulled (`ollama pull {self.model_name}`). "
                f"Details: {e}"
            )

        t_llm = time.perf_counter() - t_llm_start

        return {
            "query": user_query,
            "target_step": target_step,
            "retrieval_latency_ms": round(t_retrieval * 1000, 2),
            "llm_latency_ms": round(t_llm * 1000, 2),
            "retrieved_frames": retrieved_frames,
            "constructed_prompt": prompt,
            "explanation": explanation_text
        }