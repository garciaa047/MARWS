"""Training configuration for MARWS SAC + HER."""

from torch import nn


def get_sac_config():
    """Return SAC + HER kwargs for goal-conditioned training.

    Architecture decisions:
    - SAC (not TD3/DDPG): entropy-regularized exploration avoids premature
      convergence; auto-tuned entropy coefficient eliminates a fragile
      hyperparameter. State-of-the-art on Fetch benchmarks with HER.
    - HER with "future" strategy: relabels failed trajectories as successes
      by substituting the achieved goal for the desired goal. This is the
      original HER paper's approach and the most sample-efficient.
    - n_sampled_goal=4: generates 4 synthetic goals per real transition.
      This 5x multiplier is the standard from the HER paper.
    - Dense reward: -||achieved_goal - desired_goal|| (negative Euclidean
      distance). Provides smooth critic gradients at every state. Combined
      with HER relabeling, gives both continuous gradient signal and
      sample-efficient goal exploration. Matches Fetch reference.

    Hyperparameter rationale:
    - gamma=0.98: effective horizon of ~50 steps (0.98^50 = 0.36). Longer
      than the previous 0.95 to improve credit assignment for the full
      multi-stage task (approach -> grasp -> lift -> transport -> place).
    - tau=0.005: soft update coefficient. Standard for SAC.
    - learning_rate=1e-3: standard for SAC + robotics. Higher than PPO's
      3e-4 because off-policy algorithms are more stable with larger LR.
    - batch_size=256: standard for SAC. Larger batches give more stable
      gradients but slower per-update.
    - buffer_size=1M: large replay buffer prevents catastrophic forgetting
      of early experiences. HER fills it quickly (5x per real transition).
    - learning_starts=5000: collect 50 episodes of random data before
      training begins. More initial diversity improves HER relabeling.
    - net_arch=[256, 256]: 2 hidden layers. Matches Fetch reference
      implementations. The IK is handled analytically, so the NN only
      needs to learn a high-level spatial policy. Faster on CPU.
    - train_freq=2: train every 2 env steps. Batches NN operations for
      better CPU cache utilization. HER's 5x replay multiplier ensures
      sufficient gradient diversity.
    """
    return dict(
        # --- SAC hyperparameters ---
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=1_000_000,
        gamma=0.98,
        tau=0.005,
        ent_coef="auto",         # Auto-tuned entropy (SAC's key advantage)
        learning_starts=5000,
        train_freq=2,            # Train every 2 env steps (CPU optimization)
        gradient_steps=1,        # One gradient step per train call

        # --- Network architecture ---
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=nn.ReLU,
        ),

        # --- HER replay buffer ---
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
    )
