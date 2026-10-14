import time
import torch
import numpy as np
from typing import List
from stable_baselines3.common.callbacks import BaseCallback
from telemetry.schema import TelemetryFrame, ACTION_NAMES

# Unified with gym_wrapper.py reward shaping threshold (40% HP)
LOW_HEALTH_THRESHOLD = 0.40


class TelemetryRecorderCallback(BaseCallback):
    """
    Extracts intermediate actor-critic signals:
    - V(s): Critic scalar value estimate
    - pi(a|s): Categorical action distribution
    - Latent vector: 128-d representation from penultimate layer
    Safely supports both single and vectorized environments.
    """
    def __init__(self, record_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.record_interval = record_interval
        self.recorded_frames: List[TelemetryFrame] = []

    def extract_and_record(
        self,
        model,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        step_idx: int
    ) -> TelemetryFrame:
        """
        Core signal extraction logic usable during training or manual rollouts.
        Ensures single-sample tensor shape (1, obs_dim) so .item() never fails.
        """
        raw_obs = np.asarray(obs, dtype=np.float32)
        if raw_obs.ndim > 1:
            raw_obs = raw_obs[0]
        single_obs_batch = raw_obs.reshape(1, -1)

        obs_tensor, _ = model.policy.obs_to_tensor(single_obs_batch)

        with torch.no_grad():
            features = model.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)

            # Critic State-Value Estimate V(s) - shape (1, 1)
            value_tensor = model.policy.value_net(latent_vf)
            value_estimate = float(value_tensor.reshape(-1)[0].detach().cpu().item())

            # Actor Probability Distribution pi(a|s) - shape (1, 7)
            distribution = model.policy._get_action_dist_from_latent(latent_pi)
            probs_tensor = distribution.distribution.probs.reshape(-1)
            probs_list = probs_tensor.detach().cpu().numpy().tolist()

            # Penultimate Latent Vector - shape (1, 128)
            latent_vector = latent_pi.reshape(-1).detach().cpu().numpy().tolist()

        action_probs_map = {
            ACTION_NAMES[i]: round(float(probs_list[i]), 6)
            for i in range(len(ACTION_NAMES))
        }

        flags = {
            "is_low_health": bool(raw_obs[5] < LOW_HEALTH_THRESHOLD),
            "has_los": bool(raw_obs[9] > 0.5),
            "in_cover": bool(raw_obs[12] > 0.5)
        }

        frame = TelemetryFrame(
            timestamp=time.time(),
            step=int(step_idx),
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

        obs_batch = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        actions = self.locals["actions"]

        # Iterate over all environments in VecEnv (works for 1 or N envs)
        num_envs = obs_batch.shape[0] if obs_batch.ndim > 1 else 1
        for env_idx in range(num_envs):
            single_obs = obs_batch[env_idx] if obs_batch.ndim > 1 else obs_batch
            self.extract_and_record(
                model=self.model,
                obs=single_obs,
                action=int(actions[env_idx]),
                reward=float(rewards[env_idx]),
                done=bool(dones[env_idx]),
                step_idx=self.num_timesteps
            )
        return True