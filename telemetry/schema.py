from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time

ACTION_NAMES: Dict[int, str] = {
    0: "Move North",
    1: "Move South",
    2: "Move East",
    3: "Move West",
    4: "Shoot",
    5: "Move to Cover",
    6: "Retreat"
}

@dataclass
class TelemetryFrame:
    timestamp: float
    step: int
    observation: List[float]
    chosen_action: int
    chosen_action_name: str
    action_probabilities: Dict[str, float]  # Maps each action name to its probability pi(a|s)
    value_estimate: float                   # Critic scalar V(s)
    latent_vector: List[float]             # Penultimate layer representation
    reward: float
    done: bool
    flags: Dict[str, bool]                 # e.g., is_low_health, in_cover, has_los
    pre_action_state: Optional[Dict[str, Any]] = None
    next_observation: Optional[List[float]] = None
    transition_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_natural_language_summary(self) -> str:
        """Helper to create text embeddings in Module 3."""
        top_rejected = sorted(
            [(k, v) for k, v in self.action_probabilities.items() if k != self.chosen_action_name],
            key=lambda item: item[1],
            reverse=True
        )[0]
        
        return (
            f"Step {self.step}: Selected action '{self.chosen_action_name}' "
            f"(probability: {self.action_probabilities[self.chosen_action_name]:.2f}). "
            f"Alternative action '{top_rejected[0]}' had probability {top_rejected[1]:.2f}. "
            f"State value estimate V(s) was {self.value_estimate:.2f}. "
            f"Health flag low: {self.flags.get('is_low_health', False)}, "
            f"in cover: {self.flags.get('in_cover', False)}, "
            f"line of sight: {self.flags.get('has_los', False)}."
        )
