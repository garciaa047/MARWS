"""MARWS Environment - Simplified reward structure for reliable training."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class MarwsEnv(gym.Env):
    """Single robot arm picking a package and placing it on a platform.

    Simplified reward structure with only 3 components:
    1. Distance shaping toward current goal
    2. Grasp bonus (one-time)
    3. Delivery bonus (one-time)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=2500):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("simulation/assets/warehouse.xml")
        self.data = mujoco.MjData(self.model)

        # Observation space:
        # - joint_positions: 6
        # - joint_velocities: 6
        # - gripper_state: 1
        # - gripper_holding: 1
        # - package_position: 3
        # - bin_position: 3
        # - gripper_to_package_vector: 3
        # - package_to_bin_vector: 3
        obs_dim = 6 + 6 + 1 + 1 + 3 + 3 + 3 + 3  # = 26

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 6 joint velocities + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Core state
        self.viewer = None
        self.gripper_holding = False
        self.package_delivered = False
        self.package_fell = False

        # Reward state (minimal)
        self.grasped_once = False
        self.prev_dist_to_goal = None
        self.prev_gripper_holding = False

        # Magnetic gripper state
        self.pkg_attached = False
        self.pkg_rel_pos = None
        self.pkg_joint_qpos_adr = None
        self.contact_frames = 0
        self.no_contact_frames = 0
        self.CONTACT_THRESHOLD = 2
        self.RELEASE_THRESHOLD = 8

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Set initial robot pose (oriented toward package location)
        self.data.qpos[0] = 0.6   # joint1 - rotated toward package
        self.data.qpos[1] = -0.3  # joint2 - shoulder slightly raised
        self.data.qpos[2] = 0.8   # joint3 - elbow bent
        self.data.qpos[3] = 0.5   # joint4 - wrist
        self.data.qpos[4] = 0.0   # joint5
        self.data.qpos[5] = 0.0   # joint6

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Reset core state
        self.gripper_holding = False
        self.package_delivered = False
        self.package_fell = False
        self.current_step = 0

        # Reset reward state
        self.grasped_once = False
        self.prev_dist_to_goal = None
        self.prev_gripper_holding = False

        # Reset magnetic gripper state
        self.pkg_attached = False
        self.pkg_rel_pos = None
        self.contact_frames = 0
        self.no_contact_frames = 0

        # Get package freejoint qpos address
        pkg_joint_id = self.model.joint("package_0_joint").id
        self.pkg_joint_qpos_adr = self.model.jnt_qposadr[pkg_joint_id]

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1

        # Apply actions to actuators
        max_vel = 2.0
        self.data.ctrl[:6] = action[:6] * max_vel

        # Gripper control (two actuators, both receive same command)
        # action=-1 → closed (cmd=0.04), action=+1 → open (cmd=0)
        gripper_cmd = (1 - action[6]) / 2 * 0.04
        self.data.ctrl[6] = gripper_cmd  # Left finger
        self.data.ctrl[7] = gripper_cmd  # Right finger

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Apply magnetic gripper: lock package position to gripper when attached
        if self.pkg_attached and self.pkg_rel_pos is not None:
            gripper_pos = self._get_gripper_position()
            target_pkg_pos = gripper_pos + self.pkg_rel_pos
            self.data.qpos[self.pkg_joint_qpos_adr:self.pkg_joint_qpos_adr+3] = target_pkg_pos
            mujoco.mj_forward(self.model, self.data)

        # Update gripper holding state
        self._update_gripper_holding()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self.package_delivered or self.package_fell
        truncated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_package_position(self):
        body_id = self.model.body("package_0").id
        return self.data.xpos[body_id].copy()

    def _get_bin_position(self):
        body_id = self.model.body("bin_A").id
        return self.data.xpos[body_id].copy()

    def _get_gripper_position(self):
        gripper_body_id = self.model.body("gripper_base").id
        return self.data.xpos[gripper_body_id].copy()

    def _compute_reward(self):
        """Simplified reward: distance shaping + grasp bonus + delivery bonus.

        Reward budget (approximate):
          Phase 1 shaping (approach):  ~5.0 * 0.3m  =  1.5
          Grasp bonus:                                 20
          Phase 2 shaping (carry):    ~25.0 * 0.45m = 11.3
          Delivery bonus:                             500
          ------------------------------------------------
          Full completion total:                     ~533

          Grasp-only total:            20 - 30 (drop) = -10
          This ensures completing the task >> just grasping.
        """
        reward = 0.0
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        # Check if package is above platform
        dx_to_bin = abs(package_pos[0] - bin_pos[0])
        dy_to_bin = abs(package_pos[1] - bin_pos[1])
        above_platform = dx_to_bin < 0.15 and dy_to_bin < 0.15

        gripper_open_amt = 1.0 - (self.data.qpos[6] / 0.04)

        if not self.gripper_holding:
            # PHASE 1: Move gripper toward package
            dist = np.linalg.norm(gripper_pos - package_pos)
            if self.prev_dist_to_goal is not None:
                reward += 5.0 * (self.prev_dist_to_goal - dist)
            self.prev_dist_to_goal = dist

        else:
            # PHASE 2: Move package toward platform (or release if above)
            # Use XY distance only - so lifting isn't penalized
            dist = np.linalg.norm(package_pos[:2] - bin_pos[:2])

            if above_platform:
                # Reward opening gripper to release
                reward += 1.0 * gripper_open_amt
            else:
                # Reward moving closer to platform - STRONG shaping
                if self.prev_dist_to_goal is not None:
                    reward += 25.0 * (self.prev_dist_to_goal - dist)
                self.prev_dist_to_goal = dist

        # ONE-TIME: Grasp bonus (small - just a stepping stone, not the goal)
        if self.gripper_holding and not self.grasped_once:
            reward += 20.0
            self.grasped_once = True
            # Initialize phase 2 distance tracking immediately (XY only)
            self.prev_dist_to_goal = np.linalg.norm(package_pos[:2] - bin_pos[:2])

        # ONE-TIME: Delivery bonus (flat - stable for training)
        if self._check_package_in_bin():
            reward += 500.0
            self.package_delivered = True

        # PENALTY: Dropped package (not above platform and not delivered)
        if self.prev_gripper_holding and not self.gripper_holding:
            if not self.package_delivered and not above_platform:
                reward -= 30.0

        # PENALTY: Package fell off table
        if package_pos[2] < 0:
            reward -= 50.0
            self.package_fell = True

        # Track for next step
        self.prev_gripper_holding = self.gripper_holding

        return float(reward)

    def _check_package_in_bin(self):
        """Check if package is on the delivery platform."""
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        dx = abs(package_pos[0] - bin_pos[0])
        dy = abs(package_pos[1] - bin_pos[1])
        dz = package_pos[2] - bin_pos[2]

        # Package within platform XY bounds and resting on top
        return dx < 0.12 and dy < 0.12 and 0.01 < dz < 0.15

    def _update_gripper_holding(self):
        """Update gripper holding with magnetic gripper (programmatic attachment)."""
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()

        # Check if package is within gripper grasp zone
        dx = abs(package_pos[0] - gripper_pos[0])
        dy = abs(package_pos[1] - gripper_pos[1])
        dz = package_pos[2] - gripper_pos[2]

        package_in_grasp_zone = dx < 0.03 and dy < 0.03 and -0.06 < dz < 0.02
        gripper_closed_enough = self.data.qpos[6] > 0.025
        gripper_open = self.data.qpos[6] < 0.015

        # ATTACH: If grip conditions met and not already attached
        if not self.pkg_attached:
            if package_in_grasp_zone and gripper_closed_enough:
                self.contact_frames += 1
                self.no_contact_frames = 0

                if self.contact_frames >= self.CONTACT_THRESHOLD:
                    self.pkg_attached = True
                    self.gripper_holding = True
                    self.pkg_rel_pos = package_pos - gripper_pos
            else:
                self.no_contact_frames += 1
                self.contact_frames = 0

        # DETACH: If gripper opens while attached
        else:
            if gripper_open:
                self.no_contact_frames += 1
                if self.no_contact_frames >= self.RELEASE_THRESHOLD:
                    self.pkg_attached = False
                    self.gripper_holding = False
                    self.pkg_rel_pos = None
            else:
                self.no_contact_frames = 0

    def _get_observation(self):
        obs = []

        # Joint positions (6)
        obs.extend(self.data.qpos[:6])

        # Joint velocities (6)
        obs.extend(self.data.qvel[:6])

        # Gripper state (1)
        gripper_pos = self.data.qpos[6]
        obs.append(gripper_pos / 0.04)

        # Gripper holding (1)
        obs.append(1.0 if self.gripper_holding else 0.0)

        # Package position (3)
        package_pos = self._get_package_position()
        obs.extend(package_pos)

        # Bin position (3)
        bin_pos = self._get_bin_position()
        obs.extend(bin_pos)

        # Gripper-to-package vector (3)
        gripper_pos_world = self._get_gripper_position()
        gripper_to_pkg = package_pos - gripper_pos_world
        obs.extend(gripper_to_pkg)

        # Package-to-bin vector (3)
        pkg_to_bin = bin_pos - package_pos
        obs.extend(pkg_to_bin)

        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
