"""Evaluate a trained policy with visualization."""
import argparse
import time
import os

import ray
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from simulation.env import MarwsEnv


def env_creator(env_config):
    """Create the MARWS environment."""
    return MarwsEnv(**env_config)


def evaluate(checkpoint_path="models/staged", num_episodes=5, render=True, speed=1.0, seed=None, deterministic=False):
    """Evaluate a trained policy.

    Args:
        checkpoint_path: Path to checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render visualization
        speed: Playback speed multiplier
        seed: Random seed for reproducibility
        deterministic: If True, use deterministic actions
    """
    ray.init(ignore_reinit_error=True)
    tune.register_env("Marws-v0", env_creator)

    algo = PPO.from_checkpoint(checkpoint_path)
    env = env_creator({"render_mode": "human" if render else None})

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
            action = algo.compute_single_action(obs, explore=not deterministic)
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
                time.sleep(0.05 / speed)  # 20 Hz control

        total_rewards.append(episode_reward)
        stage_counts[highest_stage] += 1

        success_str = "SUCCESS!" if info.get("success", False) else ""
        print(f"Episode {ep+1}: reward={episode_reward:.3f}, steps={steps}, stage={highest_stage} {success_str}")

    env.close()
    algo.stop()
    ray.shutdown()

    # Summary
    avg_reward = sum(total_rewards) / len(total_rewards)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/staged", help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions")
    args = parser.parse_args()

    evaluate(os.path.abspath(args.checkpoint), args.episodes, not args.no_render,
             args.speed, args.seed, args.deterministic)
