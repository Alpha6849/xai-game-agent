import os
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from env.gym_wrapper import GymArenaEnv
from agent.models import create_ppo_agent

def train(total_timesteps: int = 200_000, checkpoint_freq: int = 50_000, model_dir: str = "./checkpoints/"):
    os.makedirs(model_dir, exist_ok=True)
    tensorboard_dir = "./tensorboard_logs/"
    os.makedirs(tensorboard_dir, exist_ok=True)

    # 1. Instantiate wrapped environment
    env = DummyVecEnv([lambda: GymArenaEnv()])

    # 2. Initialize PPO agent
    model = create_ppo_agent(env=env, tensorboard_log=tensorboard_dir)

    # 3. Setup Checkpoint Callback for skill comparison
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=model_dir,
        name_prefix="ppo_combat_agent",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    print(f"Starting training for {total_timesteps} steps...")
    print(f"Checkpoints will be saved to: {model_dir}")
    print(f"TensorBoard telemetry directed to: {tensorboard_dir}")

    # 4. Train the policy
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 5. Save finalized policy weights
    final_path = os.path.join(model_dir, "ppo_combat_agent_final.zip")
    model.save(final_path)
    print(f"Training complete. Final weights saved to {final_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Tactical Agent")
    parser.add_argument("--steps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Timesteps per checkpoint")
    args = parser.parse_args()

    train(total_timesteps=args.steps, checkpoint_freq=args.checkpoint_freq)