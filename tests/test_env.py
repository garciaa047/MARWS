"""Tests for the MARWS environment."""
import pytest
import numpy as np

from simulation.env import MarwsEnv


@pytest.fixture
def env():
    e = MarwsEnv()
    yield e
    e.close()


class TestMarwsEnvBasic:
    """Test basic environment structure."""

    def test_env_can_be_created(self, env):
        assert env is not None

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (23,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (8,)

    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset()
        assert obs.shape == (23,)
        assert isinstance(info, dict)

    def test_observation_in_observation_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_step_returns_correct_tuple(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (23,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestMarwsEnvRewards:
    """Test reward function."""

    def test_step_returns_float_reward(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_staged_rewards_returns_five_values(self, env):
        env.reset()
        staged = env.staged_rewards()
        assert len(staged) == 5


class TestPackageTracking:
    """Test package state tracking."""

    def test_package_has_position(self, env):
        env.reset()
        pos = env._get_package_position()
        assert pos.shape == (3,)

    def test_bin_has_position(self, env):
        env.reset()
        pos = env._get_bin_position()
        assert pos.shape == (3,)

    def test_gripper_has_position(self, env):
        env.reset()
        pos = env._get_gripper_position()
        assert pos.shape == (3,)


class TestEpisodeTermination:
    """Test episode termination logic."""

    def test_episode_truncated_after_max_steps(self):
        env = MarwsEnv(max_steps=10)
        env.reset()
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.zeros(8))
        assert truncated is True
        env.close()

    def test_episode_terminates_on_delivery(self, env):
        env.reset()
        env.package_delivered = True
        _, _, terminated, _, _ = env.step(np.zeros(8))
        assert terminated is True


class TestObservation:
    """Test observation structure."""

    def test_observation_contains_joint_positions(self, env):
        obs, _ = env.reset()
        joint_pos = obs[:7]
        assert len(joint_pos) == 7

    def test_observation_contains_joint_velocities(self, env):
        obs, _ = env.reset()
        joint_vel = obs[7:14]
        assert len(joint_vel) == 7

    def test_observation_contains_gripper_state(self, env):
        obs, _ = env.reset()
        gripper_state = obs[14]
        gripper_holding = obs[15]
        assert 0 <= gripper_state <= 1
        assert gripper_holding in [0.0, 1.0]

    def test_observation_contains_gripper_to_package(self, env):
        obs, _ = env.reset()
        gripper_to_pkg = obs[16:19]
        assert len(gripper_to_pkg) == 3

    def test_observation_contains_package_to_bin(self, env):
        obs, _ = env.reset()
        pkg_to_bin = obs[19:22]
        assert len(pkg_to_bin) == 3

    def test_observation_contains_height_above_table(self, env):
        obs, _ = env.reset()
        height_above_table = obs[22]
        # Package starts above the table surface
        assert height_above_table > 0


class TestSB3Compatibility:
    """Test Stable Baselines3 compatibility."""

    def test_sb3_check_env(self):
        from stable_baselines3.common.env_checker import check_env
        env = MarwsEnv()
        check_env(env, warn=True)
        env.close()
