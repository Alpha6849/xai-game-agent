import time
import torch
import numpy as np
from typing import List, Optional
from stable_baselines3.common.callbacks import BaseCallback
from telemetry.schema import TelemetryFrame, ACTION_NAMES

class TelemetryRecorderCallback(BaseCallback):
    """
    Extracts intermediate actor-critic signals:
    - V(s): Critic scalar value estimate
    - pi(a|s): Categorical action distribution
    - Latent vector: 128-d representation from shared penultimate layer
    """
    def __init__(self, record_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.record_interval = record_interval
        self.recorded_frames: List[TelemetryFrame] = []
        self._manual_step: int = 0

    def extract_and_record(
        self,
        model,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        step_idx: int
    ) -> TelemetryFrame:
        """Core signal extraction logic usable during training or manual rollouts."""
        obs_tensor, _ = model.policy.obs_to_tensor(obs)

        with torch.no_grad():
            # 1. Forward pass through feature extractor and shared layers
            features = model.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)

            # 2. Extract Critic State-Value Estimate V(s)
            value_tensor = model.policy.value_net(latent_vf)
            value_estimate = float(value_tensor.squeeze().detach().cpu().item())

            # 3. Extract Actor Probability Distribution pi(a|s)
            distribution = model.policy._get_action_dist_from_latent(latent_pi)
            probs_tensor = distribution.distribution.probs.squeeze()
            probs_list = probs_tensor.detach().cpu().numpy().tolist()

            # 4. Extract Latent Feature Vector (shared penultimate representation)
            latent_vector = latent_pi.squeeze().detach().cpu().numpy().tolist()

        action_probs_map = {
            ACTION_NAMES[i]: round(float(probs_list[i]), 4)
            for i in range(len(ACTION_NAMES))
        }

        raw_obs = obs[0] if obs.ndim > 1 else obs
        flags = {
            "is_low_health": bool(raw_obs[5] < 0.3),
            "has_los": bool(raw_obs[9] > 0.5),
            "in_cover": bool(raw_obs[12] > 0.5)
        }

        frame = TelemetryFrame(
            timestamp=time.time(),
            step=step_idx,
            observation=[float(x) for x in raw_obs],
            chosen_action=int(action),
            chosen_action_name=ACTION_NAMES[int(action)],
            action_probabilities=action_probs_map,
            value_estimate=round(value_estimate, 4),
            latent_vector=latent_vector,
            reward=float(reward),
            done=bool(done),
            flags=flags
        )

        self.recorded_frames.append(frame)
        return frame

    def _on_step(self) -> bool:
        if self.n_calls % self.record_interval != 0:
            return True

        obs = self.locals["new_obs"]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        action = int(self.locals["actions"][0])

        self.extract_and_record(
            model=self.model,
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            step_idx=self.num_timesteps
        )
        return True