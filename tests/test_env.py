"""Tests for the MARWS environment."""
import pytest
import numpy as np


class TestMarwsEnvBasic:
    """Test basic environment structure."""

    def test_env_can_be_created(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        assert env is not None

    def test_env_has_observation_space(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        assert env.observation_space is not None
        assert env.observation_space.shape == (20,)

    def test_env_has_action_space(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        assert env.action_space is not None
        assert env.action_space.shape == (7,)

    def test_env_reset_returns_observation(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        obs, info = env.reset()
        assert obs is not None
        assert obs.shape == (20,)
        assert isinstance(info, dict)

    def test_env_step_returns_correct_tuple(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info


class TestMarwsEnvRewards:
    """Test reward function."""

    def test_step_returns_float_reward(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_time_penalty_applied(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        # Zero action
        zero_action = np.zeros(7)
        _, reward, _, _, _ = env.step(zero_action)
        # Should have time penalty of -0.001
        assert reward < 0


class TestPackageTracking:
    """Test package state tracking."""

    def test_package_has_position(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        pos = env._get_package_position()
        assert pos.shape == (3,)

    def test_bin_has_position(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        pos = env._get_bin_position()
        assert pos.shape == (3,)

    def test_gripper_has_position(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        pos = env._get_gripper_position()
        assert pos.shape == (3,)


class TestEpisodeTermination:
    """Test episode termination logic."""

    def test_episode_truncated_after_max_steps(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv(max_steps=10)
        env.reset()
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.zeros(7))
        assert truncated is True

    def test_episode_terminates_on_delivery(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        # Manually set package as delivered
        env.package_delivered = True
        _, _, terminated, _, _ = env.step(np.zeros(7))
        assert terminated is True


class TestGripperContact:
    """Test gripper contact detection."""

    def test_gripper_contact_detection_exists(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        contacts = env._get_gripper_contacts()
        assert isinstance(contacts, list)

    def test_gripper_holding_initially_false(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        env.reset()
        assert env.gripper_holding is False


class TestObservation:
    """Test observation structure."""

    def test_observation_contains_joint_positions(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        obs, _ = env.reset()
        # First 6 values are joint positions
        joint_pos = obs[:6]
        assert len(joint_pos) == 6

    def test_observation_contains_joint_velocities(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        obs, _ = env.reset()
        # Values 6-12 are joint velocities
        joint_vel = obs[6:12]
        assert len(joint_vel) == 6

    def test_observation_contains_gripper_state(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        obs, _ = env.reset()
        # Value 12 is gripper state, value 13 is gripper holding
        gripper_state = obs[12]
        gripper_holding = obs[13]
        assert 0 <= gripper_state <= 1
        assert gripper_holding in [0.0, 1.0]

    def test_observation_contains_package_position(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        obs, _ = env.reset()
        # Values 14-17 are package position
        package_pos = obs[14:17]
        assert len(package_pos) == 3

    def test_observation_contains_bin_position(self):
        from simulation.env import MarwsEnv
        env = MarwsEnv()
        obs, _ = env.reset()
        # Values 17-20 are bin position
        bin_pos = obs[17:20]
        assert len(bin_pos) == 3
