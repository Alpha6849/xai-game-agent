import time
import uuid
import argparse
from stable_baselines3 import PPO
from env.gym_wrapper import CombatArenaEnv
from agent.callbacks import TelemetryRecorderCallback
from telemetry.logger import TelemetryLogger


def record_gameplay(model_path: str, episodes: int = 3, render: bool = False):
    render_mode = "human" if render else None
    env = CombatArenaEnv(render_mode=render_mode)
    model = PPO.load(model_path)
    logger = TelemetryLogger()
    callback = TelemetryRecorderCallback()

    print(f"Recording evaluation session for model: {model_path}")
    global_step = 0

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=False)

                # Extract internal signals BEFORE environment state transition occurs
                callback.extract_and_record(
                    model=model,
                    obs=obs,
                    action=int(action),
                    reward=0.0,
                    done=False,
                    step_idx=global_step
                )

                next_obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated

                # Update the last recorded frame with true transition outcome
                callback.recorded_frames[-1].reward = float(reward)
                callback.recorded_frames[-1].done = bool(done)

                if render:
                    env.render()

                obs = next_obs
                ep_step += 1
                global_step += 1

            print(
                f"Episode {ep + 1}/{episodes} concluded at episode step {ep_step} "
                f"(Cumulative steps: {global_step})"
            )

        logger.log_batch(callback.recorded_frames)
        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        saved_file = logger.flush_to_json(session_id=session_id)
        print(f"Session telemetry successfully generated: {saved_file}")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ground-Truth Telemetry Dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--render", action="store_true", help="Render Pygame screen")
    args = parser.parse_args()

    record_gameplay(model_path=args.model, episodes=args.episodes, render=args.render)