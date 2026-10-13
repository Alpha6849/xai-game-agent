import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from typing import Dict, Any

def get_ppo_policy_kwargs() -> Dict[str, Any]:
    """
    Configures an actor-critic MLP using the SB3 v2.0+ dictionary format.
    The actor (pi) and critic (vf) networks are explicitly defined[cite: 3].
    """
    return {
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "activation_fn": nn.ReLU
    }

def create_ppo_agent(env: VecEnv, tensorboard_log: str = "./tensorboard_logs/", learning_rate: float = 3e-4) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=get_ppo_policy_kwargs(),
        tensorboard_log=tensorboard_log,
        verbose=1
    )