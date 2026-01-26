"""Train single-agent MARWS with RLlib PPO."""
import argparse
import json
import os
import signal

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from simulation.env import MarwsEnv
from training.config import get_ppo_config

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForce quit requested, exiting immediately...")
        exit(1)
    print("\nInterrupt received, will save and exit after current iteration...")
    _shutdown_requested = True


def env_creator(env_config):
    """Create the MARWS environment."""
    return MarwsEnv(**env_config)


def train(num_iterations=100, checkpoint_dir="models/single_agent", resume=False):
    """Train the single-agent policy.

    Args:
        num_iterations: Number of training iterations
        checkpoint_dir: Directory to save checkpoints
        resume: If True, resume from existing checkpoint. If False, start fresh.
    """
    global _shutdown_requested
    _shutdown_requested = False

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)

    ray.init(ignore_reinit_error=True)

    # Register environment with Ray so workers can create it
    tune.register_env("Marws-v0", env_creator)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training state file (stores iteration count and best reward)
    state_file = os.path.join(checkpoint_dir, "training_state.json")

    # Check if we should resume from existing checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "rllib_checkpoint.json")
    if resume and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        algo = PPO.from_checkpoint(checkpoint_dir)

        # Load training state
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            start_iteration = state.get("iteration", 0)
            best_reward = state.get("best_reward", float("-inf"))
            best_saved_at = state.get("best_saved_at", start_iteration)
            print(f"Resuming from iteration {start_iteration}, best reward: {best_reward:.2f}")
        else:
            start_iteration = 0
            best_reward = float("-inf")
            best_saved_at = 0
    else:
        if resume:
            print("No checkpoint found, starting fresh training")
        config = get_ppo_config()
        algo = config.build_algo()
        start_iteration = 0
        best_reward = float("-inf")
        best_saved_at = 0

    total_iterations = start_iteration

    try:
        for i in range(num_iterations):
            result = algo.train()
            total_iterations = start_iteration + i + 1

            # Get episode return (reward) from env_runners
            env_runners = result.get("env_runners", {})
            reward = env_runners.get("episode_return_mean", 0.0)

            # Handle numpy types
            if hasattr(reward, 'item'):
                reward = reward.item()

            print(f"Iteration {total_iterations}: reward={reward:.2f}")

            # Save checkpoint only if best so far
            if reward > best_reward:
                best_reward = reward
                best_saved_at = total_iterations
                algo.save(checkpoint_dir)
                with open(state_file, "w") as f:
                    json.dump({"iteration": total_iterations, "best_reward": best_reward, "best_saved_at": best_saved_at}, f)
                print(f"  ^ New best! Checkpoint saved.")

            # Check for graceful shutdown request
            if _shutdown_requested:
                print(f"\nGraceful shutdown after iteration {total_iterations}")
                break

    finally:
        # Save final training state
        with open(state_file, "w") as f:
            json.dump({"iteration": total_iterations, "best_reward": best_reward, "best_saved_at": best_saved_at}, f)
        print(f"\nTraining complete. Iterations: {total_iterations}")
        print(f"Best reward: {best_reward:.2f} (saved at iteration {best_saved_at})")
        print(f"Checkpoint location: {os.path.abspath(checkpoint_dir)}")

        algo.stop()
        ray.shutdown()

    return checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.abspath("models/single_agent"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint instead of starting fresh")
    args = parser.parse_args()

    train(args.iterations, args.checkpoint_dir, args.resume)
