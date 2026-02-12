"""Train single-agent MARWS with Stable Baselines3 PPO and staged rewards."""
import argparse
import os
import signal

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from training.config import get_ppo_config


def make_env(rank, seed=0, monitor=False):
    """Return a callable that creates one MarwsEnv (for SubprocVecEnv)."""
    def _init():
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        if monitor:
            env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class GracefulShutdownCallback(BaseCallback):
    """Save model and stop training on Ctrl+C."""

    def __init__(self, save_dir, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self._shutdown = False
        self._prev_handler = None

    def _on_training_start(self):
        self._prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        if self._shutdown:
            print("\nForce quit requested, exiting immediately...")
            raise SystemExit(1)
        print("\nInterrupt received, will save and exit after current rollout...")
        self._shutdown = True

    def _on_step(self):
        if self._shutdown:
            path = os.path.join(self.save_dir, "interrupt_model")
            self.model.save(path)
            if self.verbose:
                print(f"Model saved to {path}.zip")
            return False  # stops training
        return True

    def _on_training_end(self):
        if self._prev_handler is not None:
            signal.signal(signal.SIGINT, self._prev_handler)


class StageTrackingCallback(BaseCallback):
    """Log the highest reward stage reached each rollout."""

    STAGES = [
        (0.9, "place"),
        (0.55, "hover"),
        (0.40, "lift/grasp"),
        (0.28, "contact"),
        (0.15, "reach"),
    ]

    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_rollout_end(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # SB3 stores episode stats in info["episode"] at episode end
        ep_rewards = []
        ep_lengths = []
        for info in infos:
            if "episode" in info:
                ep_rewards.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])

        if ep_rewards:
            mean_reward = np.mean(ep_rewards)
            mean_length = np.mean(ep_lengths)
            per_step = mean_reward / max(mean_length, 1)

            stage = "exploring"
            for threshold, name in self.STAGES:
                if per_step >= threshold:
                    stage = name
                    break

            if self.verbose:
                print(
                    f"Rollout: reward={mean_reward:.1f} "
                    f"(per_step={per_step:.3f}) [{stage}]"
                )
        return True

    def _on_step(self):
        return True


def train(timesteps, checkpoint_dir, resume):
    """Train with staged rewards using SB3 PPO.

    Args:
        timesteps: Total environment timesteps to train for.
        checkpoint_dir: Directory to save models and TensorBoard logs.
        resume: Path to a .zip model to resume from, or None.
    """
    config = get_ppo_config()

    n_envs = config.pop("n_envs")
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(checkpoint_dir, "logs")

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=vec_env, tensorboard_log=log_dir)
    else:
        model = PPO("MlpPolicy", vec_env, tensorboard_log=log_dir, verbose=0, **config)

    # Evaluation env (single, wrapped with Monitor for proper episode stats)
    eval_env = SubprocVecEnv([make_env(100, monitor=True)])

    callbacks = [
        GracefulShutdownCallback(save_dir=checkpoint_dir),
        StageTrackingCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            eval_freq=config.get("n_steps", 2500) * 10,  # every ~10 rollouts
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=config.get("n_steps", 2500) * 50,  # every ~50 rollouts
            save_path=checkpoint_dir,
            name_prefix="checkpoint",
        ),
    ]

    print(f"\n{'='*60}")
    print("MARWS Training with Staged Rewards (SB3 PPO)")
    print(f"{'='*60}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Envs: {n_envs}  |  n_steps: {model.n_steps}  |  batch_size: {model.batch_size}")
    print("Reward stages: reach(0.25) -> contact(0.35) -> grasp(0.40) -> lift(0.55) -> hover(0.75) -> place(1.0)")
    print(f"TensorBoard: tensorboard --logdir {os.path.abspath(log_dir)}")
    print(f"{'='*60}\n")

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    finally:
        latest_path = os.path.join(checkpoint_dir, "latest_model")
        model.save(latest_path)
        print(f"\nTraining complete. Model saved to {latest_path}.zip")
        print(f"Best model (from eval): {os.path.join(checkpoint_dir, 'best_model.zip')}")
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MARWS with SB3 PPO")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Total environment timesteps (default: 5M)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.abspath("models/staged"),
                        help="Directory for models and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .zip model to resume training from")
    args = parser.parse_args()

    train(args.timesteps, args.checkpoint_dir, args.resume)
