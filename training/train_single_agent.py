"""Train single-agent MARWS with RLlib PPO and staged rewards."""
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


def train(
    num_iterations=500,
    checkpoint_dir="models/staged",
    resume=False,
):
    """Train with staged rewards (no curriculum).

    Args:
        num_iterations: Number of training iterations
        checkpoint_dir: Directory to save checkpoints
        resume: If True, resume from existing checkpoint
    """
    global _shutdown_requested
    _shutdown_requested = False

    signal.signal(signal.SIGINT, _signal_handler)

    ray.init(ignore_reinit_error=True)
    tune.register_env("Marws-v0", env_creator)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training state file
    state_file = os.path.join(checkpoint_dir, "training_state.json")

    # Check if resuming
    checkpoint_file = os.path.join(checkpoint_dir, "rllib_checkpoint.json")
    if resume and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        algo = PPO.from_checkpoint(checkpoint_dir)

        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            start_iteration = state.get("iteration", 0)
            best_reward = state.get("best_reward", float("-inf"))
            print(f"Resuming from iteration {start_iteration}, best reward: {best_reward:.3f}")
        else:
            start_iteration = 0
            best_reward = float("-inf")
    else:
        if resume:
            print("No checkpoint found, starting fresh training")

        config = get_ppo_config()
        algo = config.build_algo()
        start_iteration = 0
        best_reward = float("-inf")

    total_iterations = start_iteration

    print(f"\n{'='*60}")
    print("MARWS Training with Staged Rewards")
    print(f"{'='*60}")
    print("Reward stages: reach(0.30) -> grasp(0.35) -> lift(0.5) -> hover(0.7) -> place(1.0)")
    print(f"{'='*60}\n")

    try:
        for i in range(num_iterations):
            result = algo.train()
            total_iterations = start_iteration + i + 1

            # Extract metrics
            env_runners = result.get("env_runners", {})
            reward = env_runners.get("episode_reward_mean", 0.0)
            ep_len = env_runners.get("episode_len_mean", 1000.0)

            # Handle numpy types
            if hasattr(reward, 'item'):
                reward = reward.item()
            if hasattr(ep_len, 'item'):
                ep_len = ep_len.item()

            # Compute per-step reward for stage detection
            per_step = reward / max(ep_len, 1)

            # Determine stage from per-step reward
            if per_step >= 0.9:
                stage = "place"
            elif per_step >= 0.5:
                stage = "hover"
            elif per_step >= 0.35:
                stage = "lift/grasp"
            elif per_step >= 0.20:
                stage = "reach"
            else:
                stage = "exploring"

            print(f"Iter {total_iterations}: reward={reward:.1f} (per_step={per_step:.3f}) [{stage}]")

            # Save checkpoint if best so far
            if reward > best_reward:
                best_reward = reward
                algo.save(checkpoint_dir)
                with open(state_file, "w") as f:
                    json.dump({
                        "iteration": total_iterations,
                        "best_reward": best_reward,
                    }, f)
                print(f"  ^ New best! Checkpoint saved.")

            # Check for graceful shutdown
            if _shutdown_requested:
                print(f"\nGraceful shutdown after iteration {total_iterations}")
                break

    finally:
        # Save final state
        with open(state_file, "w") as f:
            json.dump({
                "iteration": total_iterations,
                "best_reward": best_reward,
            }, f)

        print(f"\nTraining complete. Iterations: {total_iterations}")
        print(f"Best reward: {best_reward:.3f}")
        print(f"Checkpoint location: {os.path.abspath(checkpoint_dir)}")

        algo.stop()
        ray.shutdown()

    return checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.abspath("models/staged"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    args = parser.parse_args()

    train(
        args.iterations,
        args.checkpoint_dir,
        args.resume,
    )
