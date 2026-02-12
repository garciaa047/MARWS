"""Training configuration for MARWS."""

from torch import nn


def get_ppo_config():
    """Return SB3 PPO kwargs for staged reward training.

    Tuned for reward range [0, 1] with tanh-based shaping.
    """
    return dict(
        # Rollout
        n_steps=2500,
        n_envs=4,  # not a PPO kwarg; used by train script to set up SubprocVecEnv

        # Mini-batch SGD
        batch_size=250,  # Must evenly divide n_steps * n_envs (2500 * 4 = 10000)
        n_epochs=5,

        # Learning rate
        learning_rate=3e-4,

        # Discount / GAE
        gamma=0.99,
        gae_lambda=0.95,

        # PPO clipping
        clip_range=0.2,
        clip_range_vf=1.0,

        # Loss coefficients
        vf_coef=0.5,
        ent_coef=0.02,

        # Gradient clipping
        max_grad_norm=0.5,

        # Network architecture
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=nn.Tanh,
        ),
    )
