"""MARWS Simulation Package."""
from gymnasium.envs.registration import register

register(
    id="Marws-v0",
    entry_point="simulation.env:MarwsEnv",
    max_episode_steps=1000,
)
