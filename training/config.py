"""Training configuration for MARWS."""

from ray.rllib.algorithms.ppo import PPOConfig


def get_ppo_config():
    """Build PPO config with proper API settings to avoid deprecation warnings."""
    config = (
        PPOConfig()
        # Disable new API stack to use legacy config style
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="Marws-v0", env_config={"max_steps": 2500})
        .framework("torch")
        .env_runners(
            num_env_runners=4,
        )
        .training(
            train_batch_size=10000,
            minibatch_size=128,
            num_epochs=5,
            # Learning rate: aggressive early, then decay fast to lock in learning
            lr_schedule=[
                [0, 3e-4],           # Start high for fast learning
                [800000, 1e-4],      # Decay after ~80 iterations
                [1500000, 3e-5],     # Low for stability
                [3000000, 1e-5],     # Very low
            ],
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,  # Back to original for learning ability
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            # Entropy: high early, decay to lock in policy
            entropy_coeff_schedule=[
                [0, 0.03],
                [800000, 0.005],     # Drop faster after learning phase
                [1500000, 0.001],    # Very low to prevent drift
            ],
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
        )
    )
    return config
