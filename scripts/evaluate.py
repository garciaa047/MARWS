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


def evaluate(checkpoint_path="models\single_agent", num_episodes=1, render=True, speed=1.0, seed=None, deterministic=False):
    """Evaluate a trained policy.

    Args:
        speed: Playback speed multiplier (2.0 = 2x faster, 0.5 = half speed)
        seed: Random seed for reproducibility (None = random each time)
        deterministic: If True, use deterministic actions (no exploration noise)
    """
    ray.init(ignore_reinit_error=True)

    # Register environment with Ray so workers can create it
    tune.register_env("Marws-v0", env_creator)

    # Load the trained algorithm
    algo = PPO.from_checkpoint(checkpoint_path)

    # Create environment for evaluation
    env = env_creator({"render_mode": "human" if render else None})

    total_rewards = []

    for ep in range(num_episodes):
        # Use seed + episode number for reproducible but varied episodes
        ep_seed = (seed + ep) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Use compute_single_action for inference
            action = algo.compute_single_action(obs, explore=not deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.02 / speed)

        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: reward={episode_reward:.2f}, steps={steps}")

    env.close()
    algo.stop()
    ray.shutdown()

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models\single_agent", help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g., 2.0 for 2x speed)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (no exploration noise)")
    args = parser.parse_args()

    evaluate(os.path.abspath(args.checkpoint), args.episodes, not args.no_render,
             args.speed, args.seed, args.deterministic)
