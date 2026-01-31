"""Training configuration for MARWS."""

from ray.rllib.algorithms.ppo import PPOConfig


def get_ppo_config():
    """Build PPO config for staged reward training.

    Tuned for reward range [0, 1] with tanh-based shaping.
    """
    config = (
        PPOConfig()
        # Disable new API stack to use legacy config style
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="Marws-v0",
            env_config={"max_steps": 1000}
        )
        .framework("torch")
        .env_runners(
            num_env_runners=4,
        )
        .training(
            train_batch_size=10000,
            minibatch_size=256,
            num_epochs=5,

            # Learning rate
            lr=3e-4,

            gamma=0.99,
            lambda_=0.95,

            # PPO clipping
            clip_param=0.2,
            vf_clip_param=1.0,  # Matched to reward range [0, 1]
            vf_loss_coeff=0.5,

            # Entropy for exploration
            entropy_coeff=0.01,

            # KL divergence
            kl_coeff=0.2,
            kl_target=0.02,

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
