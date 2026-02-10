"""Evaluate a trained policy with visualization."""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from simulation.env import MarwsEnv


def evaluate(model_path, num_episodes=5, render=True, speed=1.0, seed=None, deterministic=True):
    """Evaluate a trained policy.

    Args:
        model_path: Path to SB3 model .zip file.
        num_episodes: Number of episodes to run.
        render: Whether to render visualization.
        speed: Playback speed multiplier.
        seed: Random seed for reproducibility.
        deterministic: If True, use deterministic actions.
    """
    model = PPO.load(model_path)
    env = MarwsEnv(render_mode="human" if render else None)

    total_rewards = []
    stage_counts = {"reach": 0, "grasp": 0, "lift": 0, "hover": 0, "place": 0}

    print(f"\nEvaluating {num_episodes} episodes...")
    print("-" * 50)

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        done = False
        episode_reward = 0
        steps = 0
        highest_stage = "reach"

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # Track highest stage
            current_stage = info.get("highest_stage", "reach")
            stage_order = ["reach", "grasp", "lift", "hover", "place"]
            if stage_order.index(current_stage) > stage_order.index(highest_stage):
                highest_stage = current_stage

            if render:
                env.render()
                time.sleep(0.05 / speed)

        total_rewards.append(episode_reward)
        stage_counts[highest_stage] += 1

        success_str = "SUCCESS!" if info.get("success", False) else ""
        print(f"Episode {ep+1}: reward={episode_reward:.3f}, steps={steps}, stage={highest_stage} {success_str}")

    env.close()

    # Summary
    avg_reward = np.mean(total_rewards)
    print("-" * 50)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Min reward: {min(total_rewards):.3f}")
    print(f"  Max reward: {max(total_rewards):.3f}")
    print(f"\nStage distribution:")
    for stage, count in stage_counts.items():
        pct = count / num_episodes * 100
        bar = "#" * int(pct / 5)
        print(f"  {stage:6s}: {count:3d} ({pct:5.1f}%) {bar}")

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MARWS policy")
    parser.add_argument("--model", type=str, default="models/staged/best_model.zip",
                        help="Path to SB3 model .zip file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic (non-deterministic) actions")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, not args.no_render,
             args.speed, args.seed, not args.stochastic)
