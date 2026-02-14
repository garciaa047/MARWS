"""Evaluate a trained SAC+HER policy with visualization and diagnostics."""
import argparse
import os
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from simulation.env import PandaPickAndPlaceEnv


def evaluate(model_path, num_episodes=20, render=True, speed=1.0,
             seed=None, deterministic=True):
    """Evaluate a trained policy on goal-conditioned pick-and-place.

    Reports:
    - Success rate (object within 5cm of goal)
    - Lift rate (object raised > 3cm above table)
    - Mean distance to goal at episode end
    - Mean reward per episode

    Args:
        model_path: Path to SB3 SAC model .zip file.
        num_episodes: Number of evaluation episodes.
        render: Whether to render visualization.
        speed: Playback speed multiplier.
        seed: Random seed for reproducibility.
        deterministic: If True, use deterministic actions (no stochastic sampling).
    """
    # For rendering, we use the raw env (not vectorized) so MuJoCo viewer works
    raw_env = PandaPickAndPlaceEnv(render_mode="human" if render else None)

    # Load VecNormalize stats if available (for model.predict to work correctly)
    model_dir = os.path.dirname(model_path)
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

    # We need a VecEnv for model.predict with normalized observations
    vec_env = DummyVecEnv([lambda: PandaPickAndPlaceEnv()])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded normalization stats from {vecnorm_path}")
        use_normalization = True
    else:
        print("No vecnormalize.pkl found, running without observation normalization")
        use_normalization = False

    model = SAC.load(model_path, env=vec_env)

    successes = []
    lifts = []
    distances = []
    rewards = []
    max_lifts = []  # Track max object height per episode

    print(f"\nEvaluating {num_episodes} episodes...")
    print(f"Model: {model_path}")
    print("-" * 65)

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, info = raw_env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        steps = 0
        max_obj_z = obs["achieved_goal"][2]
        min_distance = float("inf")

        while not done:
            # Normalize observation if we have stats
            if use_normalization:
                obs_normalized = vec_env.normalize_obs(obs)
            else:
                obs_normalized = obs

            action, _ = model.predict(obs_normalized, deterministic=deterministic)
            obs, reward, terminated, truncated, info = raw_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            obj_z = obs["achieved_goal"][2]
            max_obj_z = max(max_obj_z, obj_z)
            min_distance = min(min_distance, info["distance"])

            if render:
                raw_env.render()
                time.sleep(0.02 / speed)  # 50 Hz control (N_SUBSTEPS=10)

        # Episode metrics
        success = info.get("is_success", 0.0)
        final_dist = info.get("distance", 1.0)
        lifted = float(max_obj_z > 0.57)  # 3cm above table height (0.54)

        successes.append(success)
        lifts.append(lifted)
        distances.append(final_dist)
        rewards.append(ep_reward)
        max_lifts.append(max_obj_z)

        status = "SUCCESS" if success else ""
        lift_str = "LIFTED" if lifted else ""
        print(
            f"Ep {ep+1:>3}: reward={ep_reward:>7.1f}  "
            f"dist={final_dist:.3f}  "
            f"min_dist={min_distance:.3f}  "
            f"max_z={max_obj_z:.3f}  "
            f"{status} {lift_str}"
        )

    raw_env.close()
    vec_env.close()

    # Summary statistics
    success_rate = np.mean(successes)
    lift_rate = np.mean(lifts)
    mean_dist = np.mean(distances)
    mean_reward = np.mean(rewards)

    print("-" * 65)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Success rate:  {success_rate:.1%} ({int(sum(successes))}/{num_episodes})")
    print(f"  Lift rate:     {lift_rate:.1%} ({int(sum(lifts))}/{num_episodes})")
    print(f"  Mean distance: {mean_dist:.3f} m")
    print(f"  Mean reward:   {mean_reward:.1f}")
    print(f"  Mean max z:    {np.mean(max_lifts):.3f} m")

    # Diagnosis
    print(f"\n--- Diagnosis ---")
    if success_rate > 0.7:
        print("  Agent is successfully completing pick-and-place!")
    elif lift_rate > 0.5:
        print("  Agent lifts the object but fails to place it accurately.")
        print("  Consider: more training steps, or check goal distribution.")
    elif lift_rate > 0.1:
        print("  Agent occasionally lifts. Grasping is partially learned.")
        print("  Consider: more training, or increase n_sampled_goal in HER.")
    elif mean_dist < 0.3:
        print("  Agent approaches but doesn't grasp. Hover-like behavior.")
        print("  This should NOT happen with dense reward + HER.")
        print("  Check: gripper control, friction, IK accuracy.")
    else:
        print("  Agent is still in early exploration.")
        print("  Expected: more training needed (>1M steps for initial progress).")

    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MARWS policy")
    parser.add_argument(
        "--model", type=str, default="models/sac_her/best_model.zip",
        help="Path to SB3 SAC model .zip file",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate(
        args.model, args.episodes, not args.no_render,
        args.speed, args.seed, not args.stochastic,
    )
