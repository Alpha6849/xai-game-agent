import json
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

    RESPONSE_FORMAT = {
        "type": "object",
        "properties": {
            "takeaway": {
                "type": "string",
                "description": (
                    "One cautious sentence based on the target frame. State that the logs do not "
                    "establish why the action was sampled or what the unchosen action would have done."
                ),
            }
        },
        "required": ["takeaway"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        model_name: str = "qwen3:4b",
        persist_dir: str = "./data/chromadb",
        temperature: float = 0.0
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

    @staticmethod
    def _target_metadata(retrieved_frames, target_step):
        if not retrieved_frames:
            return None
        target_frame = next(
            (frame for frame in retrieved_frames
             if target_step is not None and frame.get("metadata", {}).get("step") == target_step),
            retrieved_frames[0],
        )
        return target_frame.get("metadata", {})

    @staticmethod
    def _requested_alternative(user_query, chosen_name, default_name, default_probability, probabilities):
        normalized_query = re.sub(r"[^a-z0-9]+", " ", user_query.casefold()).strip()
        for action_name, probability in probabilities.items():
            normalized_action = re.sub(r"[^a-z0-9]+", " ", action_name.casefold()).strip()
            if (action_name != chosen_name and normalized_action
                    and re.search(rf"(?<!\w){re.escape(normalized_action)}(?!\w)", normalized_query)):
                return action_name, float(probability)
        return default_name, float(default_probability)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    @staticmethod
    def _format_grounded_fallback(user_query, retrieved_frames, target_step):
        """Build a short, factual answer when the model misses the required format."""
        meta = ExplainableRAGClient._target_metadata(retrieved_frames, target_step)
        if meta is None:
            return (
                "Decision: No matching recorded decision was found.\n"
                "State: No target-frame telemetry is available.\n"
                "Limit: A logged choice cannot be explained without matching telemetry."
            )

        step = meta.get("step", target_step)
        chosen_name, chosen_probability, default_name, default_probability, probabilities = (
            CounterfactualPromptEngine.extract_action_comparison(meta)
        )
        alternative_name, alternative_probability = ExplainableRAGClient._requested_alternative(
            user_query, chosen_name, default_name, default_probability, probabilities
        )

        step_label = f"Step {step}" if step is not None else "Recorded decision"
        if alternative_name == "None":
            decision_line = (
                f"Decision: {step_label}: {chosen_name} (π(a|s)={chosen_probability:.6f}; "
                f"{chosen_probability * 100:.1f}%) was selected; "
                "no rejected alternative was recorded."
            )
        else:
            decision_line = (
                f"Decision: {step_label}: {chosen_name} (π(a|s)={chosen_probability:.6f}; "
                f"{chosen_probability * 100:.1f}%) was selected; "
                f"{alternative_name} (π(a|s)={alternative_probability:.6f}; "
                f"{alternative_probability * 100:.1f}%) was the named alternative."
            )

        def number_or_missing(value, digits=4):
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "not recorded"

        def flag_label(value, true_label, false_label):
            parsed = ExplainableRAGClient._as_bool(value)
            if parsed is None:
                return "unknown"
            return true_label if parsed else false_label

        hp = number_or_missing(meta.get("agent_hp_pct"), digits=1)
        ammo = meta.get("ammo_count", "not recorded")
        state_line = (
            f"State: V(s)={number_or_missing(meta.get('value_estimate'))}; "
            f"reward={number_or_missing(meta.get('reward'))}; HP={hp}%; ammo={ammo}/30; "
            f"{flag_label(meta.get('has_los'), 'LOS clear', 'LOS blocked')}; "
            f"{flag_label(meta.get('in_cover'), 'in cover', 'not in cover')}."
        )
        limit_line = (
            "Limit: The log records the sampled action and state; it does not establish why it was sampled "
            "or what the alternative's outcome would have been."
        )
        return "\n".join((decision_line, state_line, limit_line))

    @staticmethod
    def _compose_answer_with_takeaway(user_query, retrieved_frames, target_step, takeaway):
        if not isinstance(takeaway, str):
            return None
        takeaway = " ".join(takeaway.split()).strip().strip("`\"'")
        if takeaway.casefold().startswith("limit:"):
            takeaway = takeaway.split(":", 1)[1].strip()
        if not takeaway:
            return None

        has_outcome_limit = re.search(
            r"\b(?:outcome|result|consequence|what\s+.{1,45}\s+(?:would|could|might)|"
            r"(?:would|could|might)\s+have\s+done)\b",
            takeaway,
            re.IGNORECASE,
        )
        if not has_outcome_limit:
            if takeaway[-1] not in ".!?":
                takeaway += "."
            takeaway += " The logs also cannot establish what the unchosen action would have done."

        trusted_lines = ExplainableRAGClient._format_grounded_fallback(
            user_query, retrieved_frames, target_step
        ).splitlines()
        if len(trusted_lines) != 3:
            return None
        return "\n".join((trusted_lines[0], trusted_lines[1], f"Limit: {takeaway}"))

    @staticmethod
    def _contains_number(text, value, as_probability=False):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        candidates = {f"{number:.4f}", f"{number:g}"}
        if as_probability:
            candidates.add(f"{number * 100:.1f}%")
        return any(candidate.casefold() in text.casefold() for candidate in candidates)

    @staticmethod
    def _answer_validation_error(explanation, user_query, retrieved_frames, target_step):
        lines = [line.strip() for line in explanation.splitlines() if line.strip()]
        if (len(lines) != 3 or len(explanation.split()) > 75
                or not lines[0].startswith("Decision:")
                or not lines[1].startswith("State:")
                or not lines[2].startswith("Limit:")):
            return "The composed answer did not have the expected three-line format."

        meta = ExplainableRAGClient._target_metadata(retrieved_frames, target_step)
        if meta is None:
            return "No target-frame telemetry was available to validate against."

        step = meta.get("step", target_step)
        chosen_name, chosen_probability, default_name, default_probability, probabilities = (
            CounterfactualPromptEngine.extract_action_comparison(meta)
        )
        alternative_name, alternative_probability = ExplainableRAGClient._requested_alternative(
            user_query, chosen_name, default_name, default_probability, probabilities
        )
        decision_line, state_line, limit_line = lines

        if (step is not None
                and not re.search(rf"(?<!\d){re.escape(str(step))}(?!\d)", decision_line)):
            return "The decision line did not include the recorded step number."
        if (chosen_name.casefold() not in decision_line.casefold()
                or not ExplainableRAGClient._contains_number(decision_line, chosen_probability, True)):
            return "The decision line did not match the recorded action or its probability."
        if alternative_name != "None" and (
                alternative_name.casefold() not in decision_line.casefold()
                or not ExplainableRAGClient._contains_number(decision_line, alternative_probability, True)):
            return "The decision line did not match the requested alternative or its probability."

        value = meta.get("value_estimate")
        reward = meta.get("reward")
        if ("v(s)" not in state_line.casefold() or "reward" not in state_line.casefold()
                or not ExplainableRAGClient._contains_number(state_line, value)
                or not ExplainableRAGClient._contains_number(state_line, reward)):
            return "The state line did not match the logged value estimate or reward."

        for flag, true_label, false_label in (
            (meta.get("has_los"), "LOS clear", "LOS blocked"),
            (meta.get("in_cover"), "in cover", "not in cover"),
        ):
            parsed_flag = ExplainableRAGClient._as_bool(flag)
            expected = true_label if parsed_flag else false_label
            if parsed_flag is not None and expected.casefold() not in state_line.casefold():
                return f"The state line did not match the logged {expected} flag."

        cautious_limit = re.search(
            r"\b(?:do not|does not|did not|don't|doesn't|didn't|cannot|can't|"
            r"unable to|not enough to|insufficient to)\b.{0,60}"
            r"\b(?:establish|prove|show|tell|explain|determine|reveal|identify|record|say|specify|confirm|infer)\b"
            r".{0,40}\b(?:why|cause|reason|motive|intent)\b",
            limit_line,
            re.IGNORECASE,
        )
        reason_not_recorded = re.search(
            r"\b(?:why|cause|reason|motive|intent)\b.{0,35}"
            r"\b(?:isn't|is not|wasn't|was not|remains)\b.{0,15}"
            r"\b(?:recorded|known|clear|available|established)\b",
            limit_line,
            re.IGNORECASE,
        )
        mentions_outcome = re.search(
            r"\b(?:outcome|result|consequence|what\s+.{1,45}\s+(?:would|could|might)\s+"
            r"(?:have\s+)?(?:do|done|happen|caused)|(?:would|could|might)\s+have\s+"
            r"(?:done|happened|resulted)|predict\w*\s+.{0,25}(?:outcome|result|consequence))\b",
            limit_line,
            re.IGNORECASE,
        )
        unsupported_cause = re.search(
            r"\b(because|therefore|thus|as a result|in order to|to avoid|to maintain)\b",
            explanation,
            re.IGNORECASE,
        )
        if not (cautious_limit or reason_not_recorded):
            return "Qwen's takeaway did not clearly say that the logs leave the decision reason unknown."
        if not mentions_outcome:
            return "Qwen's takeaway did not clearly state that the unchosen action's outcome is unknown."
        if unsupported_cause:
            return "Qwen's takeaway included a causal or intent phrase that is not established by the log."
        return None

    @staticmethod
    def _is_well_formed_answer(explanation, user_query, retrieved_frames, target_step):
        """Compatibility wrapper for callers that need a boolean validation result."""
        return ExplainableRAGClient._answer_validation_error(
            explanation, user_query, retrieved_frames, target_step
        ) is None

    def answer_question(
        self,
        user_query: str,
        explicit_step: Optional[int] = None,
        window: int = 2,
        top_k: int = 3,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves relevant telemetry frames, builds a counterfactual prompt,
        queries the local Ollama model, and returns the explanation + raw evidence.
        """
        t_start = time.perf_counter()

        target_step = explicit_step if explicit_step is not None else self._extract_step_from_query(user_query)

        if target_step is not None:
            retrieved_frames = self.retriever.retrieve_by_step(
                step=target_step, window=window, session_id=session_id
            )
        else:
            where_filter = {"session_id": session_id} if session_id is not None else None
            retrieved_frames = self.retriever.retrieve_by_query(
                query=user_query, top_k=top_k, where_filter=where_filter
            )

        t_retrieval = time.perf_counter() - t_start

        prompt = self.prompt_engine.build_counterfactual_prompt(
            user_query=user_query,
            retrieved_frames=retrieved_frames,
            focus_step=target_step
        )

        t_llm_start = time.perf_counter()
        explanation_fallback = False
        fallback_reason = None
        raw_model_response = ""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_engine.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                format=self.RESPONSE_FORMAT,
                # Keep Qwen3's internal reasoning disabled for concise dashboard answers.
                think=False,
                keep_alive="10m",
                options={
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "num_predict": 96
                }
            )
            raw_response = response["message"]["content"]
            raw_model_response = raw_response if isinstance(raw_response, str) else str(raw_response)
            try:
                response_data = json.loads(raw_response)
            except (json.JSONDecodeError, TypeError):
                response_data = {}
            explanation_text = self._compose_answer_with_takeaway(
                user_query,
                retrieved_frames,
                target_step,
                response_data.get("takeaway") if isinstance(response_data, dict) else None,
            )
            fallback_reason = self._answer_validation_error(
                explanation_text, user_query, retrieved_frames, target_step
            ) if explanation_text else "Qwen did not return a usable takeaway in the expected JSON field."
            if fallback_reason:
                explanation_text = self._format_grounded_fallback(
                    user_query, retrieved_frames, target_step
                )
                explanation_fallback = True
        except Exception as e:
            explanation_text = (
                f"[Ollama Inference Error]: Could not reach local model '{self.model_name}'. "
                f"Verify Ollama is running (`ollama serve`) and model is pulled (`ollama pull {self.model_name}`). "
                f"Details: {e}"
            )
            fallback_reason = f"Ollama inference failed: {e}"

        t_llm = time.perf_counter() - t_llm_start

        return {
            "query": user_query,
            "target_step": target_step,
            "retrieval_latency_ms": round(t_retrieval * 1000, 2),
            "llm_latency_ms": round(t_llm * 1000, 2),
            "retrieved_frames": retrieved_frames,
            "constructed_prompt": prompt,
            "explanation": explanation_text,
            "explanation_fallback": explanation_fallback,
            "fallback_reason": fallback_reason,
            "raw_model_response": raw_model_response
        }
