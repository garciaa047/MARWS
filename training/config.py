"""Training configuration for MARWS."""

from ray.rllib.algorithms.ppo import PPOConfig


def get_ppo_config():
    """Build PPO config optimized to prevent policy collapse.

    Key changes to prevent "peak then collapse" behavior:
    1. Faster LR decay - lock in learning by 100 iterations
    2. Faster entropy decay - reduce exploration as policy improves
    3. Larger batch size - more stable gradient estimates
    4. More SGD epochs - extract more learning per batch
    5. KL targeting - prevent policy from changing too drastically
    """
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
            # Larger batch = more stable updates
            train_batch_size=20000,  # Increased from 10000
            minibatch_size=256,      # Increased from 128
            num_epochs=10,           # Increased from 5 - more learning per batch

            # Learning rate: decay to near-zero by iteration 125
            # At 20k batch size: 100 iters = 2M timesteps
            lr_schedule=[
                [0, 5e-4],           # Start high for fast initial learning
                [500000, 3e-4],      # ~25 iters: still learning
                [1000000, 1e-4],     # ~50 iters: slow down
                [1500000, 5e-5],     # ~75 iters: fine-tuning
                [2000000, 1e-5],     # ~100 iters: nearly locked
                [2500000, 5e-6],     # ~125 iters: frozen
            ],

            gamma=0.99,
            lambda_=0.95,  # GAE lambda

            # PPO clipping - tight for stability
            clip_param=0.15,
            vf_clip_param=1.0,       # Matched to scaled reward range [-5, 14]
            vf_loss_coeff=0.5,

            # Entropy: decay to near-zero by iteration 125
            entropy_coeff_schedule=[
                [0, 0.02],           # Start with exploration
                [500000, 0.01],      # ~25 iters: reduce
                [1000000, 0.003],    # ~50 iters: low
                [1500000, 0.001],    # ~75 iters: minimal
                [2000000, 0.0003],   # ~100 iters: near-zero
                [2500000, 0.0001],   # ~125 iters: frozen
            ],

            # KL divergence targeting - AGGRESSIVE to prevent collapse
            # Higher coeff + tighter target = stronger constraint on policy change
            kl_coeff=0.5,            # 2.5x stronger initial KL penalty
            kl_target=0.005,         # 2x tighter target (was 0.01)

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
