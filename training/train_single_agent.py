"""Train MARWS Panda pick-and-place with SAC + HER.

Key changes from the previous PPO approach:
1. SAC (off-policy) replaces PPO (on-policy) — each experience is reused
   many times, dramatically improving sample efficiency.
2. HER (Hindsight Experience Replay) — turns every failed trajectory into
   a successful one by relabeling the desired goal with the achieved goal.
3. Goal-conditioned environment — observation is a Dict with
   {observation, achieved_goal, desired_goal}. Uses MultiInputPolicy.
4. Dense reward: -||achieved - desired|| (negative Euclidean distance).
   Provides smooth critic gradients while HER handles sample efficiency.
   Matches the gymnasium_robotics FetchPickAndPlace dense formulation.
5. Cartesian action space (4-dim) — simpler than 8-dim joint space.
6. Shorter episodes (100 steps) — forces efficient behavior.
"""
import argparse
import os
import signal
from collections import deque

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.her import HerReplayBuffer

from training.config import get_sac_config

# Optimize CPU threading for small networks. On CPU, a single thread avoids
# synchronization overhead for 256-dim matrix multiplies. Benchmark with 1-4.
torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def make_env(seed=0):
    """Create a single MarwsEnv instance wrapped in Monitor."""
    def _init():
        from simulation.env import PandaPickAndPlaceEnv
        env = PandaPickAndPlaceEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


class GracefulShutdownCallback(BaseCallback):
    """Save model, normalization stats, and stop training on Ctrl+C."""

    def __init__(self, save_dir, train_env=None, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.train_env = train_env
        self._shutdown = False
        self._prev_handler = None

    def _on_training_start(self):
        self._prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        # Windows: also handle SIGBREAK (Ctrl+Break, and some IDEs)
        if hasattr(signal, "SIGBREAK"):
            self._prev_break_handler = signal.getsignal(signal.SIGBREAK)
            signal.signal(signal.SIGBREAK, self._handler)

    def _handler(self, signum, frame):
        if self._shutdown:
            print("\nForce quit requested, exiting immediately...")
            raise SystemExit(1)
        print("\nInterrupt received, will save and exit after current step...")
        self._shutdown = True

    def _on_step(self):
        if self._shutdown:
            path = os.path.join(self.save_dir, "interrupt_model")
            self.model.save(path)
            if self.train_env is not None:
                self.train_env.save(os.path.join(self.save_dir, "interrupt_vecnormalize.pkl"))
            if self.verbose:
                print(f"Model saved to {path}.zip")
            return False
        return True

    def _on_training_end(self):
        if self._prev_handler is not None:
            signal.signal(signal.SIGINT, self._prev_handler)
        if hasattr(self, "_prev_break_handler") and self._prev_break_handler is not None:
            signal.signal(signal.SIGBREAK, self._prev_break_handler)


class SuccessRateCallback(BaseCallback):
    """Log success rate and diagnostic metrics during training.

    Tracks (rolling window of last 100 episodes):
    - Episode success rate (object reached goal)
    - Mean distance to goal at episode end
    - Object lift rate (object z > initial + 3cm)
    - Mean episode reward
    - Gripper-object distance (approach progress)
    - Goal height distribution (fraction of lifted goals)
    """

    def __init__(self, log_freq=1000, window=100, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_successes = deque(maxlen=window)
        self._episode_distances = deque(maxlen=window)
        self._episode_lifts = deque(maxlen=window)
        self._episode_rewards = deque(maxlen=window)
        self._episode_goal_heights = deque(maxlen=window)
        self._episode_grip_dists = deque(maxlen=window)
        self._current_ep_reward = 0.0
        self._total_episodes = 0

    def _on_step(self):
        # Accumulate per-step reward
        rewards = self.locals.get("rewards", [])
        if len(rewards) > 0:
            self._current_ep_reward += rewards[0]

        # Check for episode end
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if dones[i] if len(dones) > i else False:
                self._episode_successes.append(info.get("is_success", 0.0))
                self._episode_distances.append(info.get("distance", 1.0))
                obj_z = info.get("object_z", 0.54)
                self._episode_lifts.append(float(obj_z > 0.57))  # 3cm above table
                self._episode_rewards.append(self._current_ep_reward)
                self._episode_goal_heights.append(info.get("goal_z", 0.54))
                self._episode_grip_dists.append(info.get("grip_obj_dist", 1.0))
                self._current_ep_reward = 0.0
                self._total_episodes += 1

        # Periodic logging
        if self.num_timesteps % self.log_freq == 0 and self._episode_successes:
            success_rate = np.mean(self._episode_successes)
            mean_dist = np.mean(self._episode_distances)
            lift_rate = np.mean(self._episode_lifts)
            mean_reward = np.mean(self._episode_rewards)
            mean_goal_z = np.mean(self._episode_goal_heights)
            lifted_goal_rate = np.mean([z > 0.57 for z in self._episode_goal_heights])
            mean_grip_dist = np.mean(self._episode_grip_dists)

            # Log to TensorBoard
            self.logger.record("marws/success_rate", success_rate)
            self.logger.record("marws/mean_distance", mean_dist)
            self.logger.record("marws/lift_rate", lift_rate)
            self.logger.record("marws/mean_episode_reward", mean_reward)
            self.logger.record("marws/mean_goal_height", mean_goal_z)
            self.logger.record("marws/lifted_goal_rate", lifted_goal_rate)
            self.logger.record("marws/mean_grip_obj_dist", mean_grip_dist)
            self.logger.record("marws/total_episodes", self._total_episodes)

            if self.verbose:
                print(
                    f"[{self.num_timesteps:>8,} steps | {self._total_episodes:>5} eps] "
                    f"success={success_rate:.1%} "
                    f"dist={mean_dist:.3f} "
                    f"grip={mean_grip_dist:.3f} "
                    f"lift={lift_rate:.1%} "
                    f"reward={mean_reward:.1f}"
                )

        return True


def train(timesteps, checkpoint_dir, resume):
    """Train with SAC + HER.

    Args:
        timesteps: Total environment timesteps.
        checkpoint_dir: Directory for models, logs, and checkpoints.
        resume: Path to .zip model to resume from, or None.
    """
    config = get_sac_config()
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(checkpoint_dir, "logs")
    vecnorm_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")

    # Training environment with observation normalization
    train_env = DummyVecEnv([make_env(seed=0)])
    if resume and os.path.exists(vecnorm_path):
        train_env = VecNormalize.load(vecnorm_path, train_env)
        print(f"Loaded normalization stats from {vecnorm_path}")
    else:
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        )

    # Separate HER kwargs from SAC kwargs (non-mutating)
    her_kwargs = config.get("replay_buffer_kwargs")
    sac_kwargs = {k: v for k, v in config.items() if k != "replay_buffer_kwargs"}

    if resume:
        print(f"Resuming from {resume}")
        model = SAC.load(resume, env=train_env, tensorboard_log=log_dir)
    else:
        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            tensorboard_log=log_dir,
            verbose=0,
            **sac_kwargs,
        )

    # Evaluation environment (uses same normalization stats but doesn't update them)
    eval_env = DummyVecEnv([make_env(seed=100)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        training=False,
    )
    # Sync normalization stats from training env
    eval_env.obs_rms = train_env.obs_rms

    callbacks = CallbackList([
        GracefulShutdownCallback(save_dir=checkpoint_dir, train_env=train_env),
        SuccessRateCallback(log_freq=5000, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            eval_freq=50_000,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=checkpoint_dir,
            name_prefix="checkpoint",
        ),
    ])

    # Print training summary
    print(f"\n{'='*65}")
    print("MARWS Training: SAC + HER (Goal-Conditioned Dense Reward)")
    print(f"{'='*65}")
    print(f"Timesteps:     {timesteps:,}")
    print(f"Algorithm:     SAC (off-policy, entropy-regularized)")
    print(f"Replay buffer: HER (n_sampled_goal=4, strategy=future)")
    print(f"Reward:        Dense (-||obj-goal||) + approach shaping")
    print(f"Action space:  4-dim Cartesian (dx, dy, dz, gripper)")
    print(f"Episodes:      100 steps ({100 * 0.02:.1f}s sim time)")
    print(f"Network:       {config['policy_kwargs']['net_arch']} ReLU")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Gamma:         {config['gamma']}")
    print(f"Buffer size:   {config['buffer_size']:,}")
    print(f"Train freq:    {config['train_freq']} steps")
    print(f"Obs normalize: VecNormalize (running mean/std)")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"TensorBoard:   tensorboard --logdir {os.path.abspath(log_dir)}")
    print(f"{'='*65}\n")

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    finally:
        latest_path = os.path.join(checkpoint_dir, "latest_model")
        model.save(latest_path)
        train_env.save(vecnorm_path)
        print(f"\nTraining complete. Model saved to {latest_path}.zip")
        print(f"Normalization stats saved to {vecnorm_path}")
        print(f"Best model: {os.path.join(checkpoint_dir, 'best_model.zip')}")
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MARWS with SAC + HER")
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Total environment timesteps (default: 2M, realistic for overnight CPU)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=os.path.abspath("models/sac_her"),
        help="Directory for models and logs",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to .zip model to resume training from",
    )
    args = parser.parse_args()

    train(args.timesteps, args.checkpoint_dir, args.resume)
