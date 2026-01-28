"""MARWS Environment - Franka Emika Panda robot with position control."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class MarwsEnv(gym.Env):
    """Panda robot arm picking a package and placing it on a platform.

    Uses position control: actions are target joint positions.
    7-DOF arm + parallel gripper.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Panda joint control ranges (from panda.xml actuators)
    JOINT_CTRL_RANGES = [
        (-2.8973, 2.8973),   # joint1
        (-1.7628, 1.7628),   # joint2
        (-2.8973, 2.8973),   # joint3
        (-3.0718, -0.0698),  # joint4
        (-2.8973, 2.8973),   # joint5
        (-0.0175, 3.7525),   # joint6
        (-2.8973, 2.8973),   # joint7
    ]

    def __init__(self, render_mode=None, max_steps=2500):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Load MuJoCo model (Panda + warehouse scene)
        self.model = mujoco.MjModel.from_xml_path("simulation/franka_emika_panda/warehouse_scene.xml")
        self.data = mujoco.MjData(self.model)

        # Cache joint and body IDs for faster access
        self._cache_ids()

        # Observation space (22-dim):
        # - joint_positions: 7 (Panda has 7 DOF)
        # - joint_velocities: 7
        # - gripper_state: 1
        # - gripper_holding: 1
        # - gripper_to_package_vector: 3
        # - package_to_bin_vector: 3
        obs_dim = 7 + 7 + 1 + 1 + 3 + 3  # = 22

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 7 joint positions + 1 gripper command (all normalized [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Core state
        self.viewer = None
        self.gripper_holding = False
        self.package_delivered = False
        self.package_fell = False

        # Reward state
        self.grasped_once = False
        self.prev_dist_to_goal = None
        self.prev_gripper_holding = False

        # Contact-based grasp detection
        self.grasp_contact_frames = 0
        self.GRASP_CONTACT_THRESHOLD = 3  # Frames of contact needed to confirm grasp

    def _cache_ids(self):
        """Cache joint and body IDs for faster runtime access."""
        # Arm joint IDs (for qpos/qvel access)
        self.arm_joint_ids = [
            self.model.joint(f"joint{i}").id for i in range(1, 8)
        ]
        self.arm_qpos_addrs = [
            self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids
        ]
        self.arm_qvel_addrs = [
            self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids
        ]

        # Finger joint (for gripper state)
        self.finger_joint_id = self.model.joint("finger_joint1").id
        self.finger_qpos_addr = self.model.jnt_qposadr[self.finger_joint_id]

        # Body IDs
        self.hand_body_id = self.model.body("hand").id
        self.package_body_id = self.model.body("package_0").id
        self.bin_body_id = self.model.body("bin_A").id

        # Package geom ID for contact detection
        self.package_geom_id = self.model.geom("package_0_geom").id

        # Finger body IDs for contact detection
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose to Panda's "home" configuration
        # From panda.xml keyframe: qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
        home_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        for i, addr in enumerate(self.arm_qpos_addrs):
            self.data.qpos[addr] = home_qpos[i]

        # Set gripper to open position
        self.data.qpos[self.finger_qpos_addr] = 0.04  # finger_joint1
        # finger_joint2 follows via equality constraint

        # Set initial control to match pose
        for i in range(7):
            self.data.ctrl[i] = home_qpos[i]
        self.data.ctrl[7] = 255  # Gripper open

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

        # Reset grasp detection state
        self.grasp_contact_frames = 0

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1

        # Map arm actions [-1, 1] to joint control ranges
        for i in range(7):
            ctrl_min, ctrl_max = self.JOINT_CTRL_RANGES[i]
            # action in [-1, 1] → ctrl in [ctrl_min, ctrl_max]
            self.data.ctrl[i] = ctrl_min + (action[i] + 1) / 2 * (ctrl_max - ctrl_min)

        # Gripper control: action[-1] in [-1, 1]
        # -1 = closed (ctrl=0), +1 = open (ctrl=255)
        self.data.ctrl[7] = (action[7] + 1) / 2 * 255

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Update gripper holding state based on physical contact
        self._update_gripper_holding()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self.package_delivered or self.package_fell
        truncated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_package_position(self):
        return self.data.xpos[self.package_body_id].copy()

    def _get_bin_position(self):
        return self.data.xpos[self.bin_body_id].copy()

    def _get_gripper_position(self):
        return self.data.xpos[self.hand_body_id].copy()

    def _get_finger_qpos(self):
        """Get finger joint position. 0 = closed, 0.04 = open."""
        return self.data.qpos[self.finger_qpos_addr]

    def _compute_reward(self):
        """Simplified reward with LOW scale for stable value function learning."""
        reward = 0.0
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        # Check if package is above platform
        dx_to_bin = abs(package_pos[0] - bin_pos[0])
        dy_to_bin = abs(package_pos[1] - bin_pos[1])
        above_platform = dx_to_bin < 0.12 and dy_to_bin < 0.12

        # Panda gripper: 0 = closed, 0.04 = open
        finger_pos = self._get_finger_qpos()
        gripper_open_amt = finger_pos / 0.04  # 0 = closed, 1 = open

        if not self.gripper_holding:
            # PHASE 1: Move gripper toward package
            dist = np.linalg.norm(gripper_pos - package_pos)
            if self.prev_dist_to_goal is not None:
                reward += 1.0 * (self.prev_dist_to_goal - dist)
            self.prev_dist_to_goal = dist

        else:
            # PHASE 2: Move package toward platform (or release if above)
            dist = np.linalg.norm(package_pos[:2] - bin_pos[:2])

            if above_platform:
                reward += 0.1 * gripper_open_amt
            else:
                if self.prev_dist_to_goal is not None:
                    reward += 5.0 * (self.prev_dist_to_goal - dist)
                self.prev_dist_to_goal = dist

        # ONE-TIME: Grasp bonus
        if self.gripper_holding and not self.grasped_once:
            reward += 1.0
            self.grasped_once = True
            self.prev_dist_to_goal = np.linalg.norm(package_pos[:2] - bin_pos[:2])

        # ONE-TIME: Delivery bonus
        if self._check_package_in_bin():
            reward += 10.0
            self.package_delivered = True

        # PENALTY: Dropped package
        if self.prev_gripper_holding and not self.gripper_holding:
            if not self.package_delivered and not above_platform:
                reward -= 2.0

        # PENALTY: Package fell
        if package_pos[2] < 0:
            reward -= 5.0
            self.package_fell = True

        self.prev_gripper_holding = self.gripper_holding
        return float(reward)

    def _check_package_in_bin(self):
        """Check if package is on the delivery platform."""
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        dx = abs(package_pos[0] - bin_pos[0])
        dy = abs(package_pos[1] - bin_pos[1])
        dz = package_pos[2] - bin_pos[2]

        return dx < 0.10 and dy < 0.10 and 0.01 < dz < 0.15

    def _update_gripper_holding(self):
        """Update gripper holding based on physical contact detection."""
        # Check if fingers are in contact with the package
        finger_touching_package = self._check_finger_package_contact()

        # Gripper must be somewhat closed to be considered holding
        finger_pos = self._get_finger_qpos()
        gripper_closed_enough = finger_pos < 0.03  # Fingers reasonably closed

        if finger_touching_package and gripper_closed_enough:
            self.grasp_contact_frames += 1
            if self.grasp_contact_frames >= self.GRASP_CONTACT_THRESHOLD:
                self.gripper_holding = True
        else:
            self.grasp_contact_frames = 0
            self.gripper_holding = False

    def _check_finger_package_contact(self):
        """Check if either finger is in contact with the package."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get body IDs for the contacting geoms
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # Check if contact involves package and a finger
            involves_package = (geom1 == self.package_geom_id or geom2 == self.package_geom_id)
            involves_finger = (body1 == self.left_finger_body_id or body1 == self.right_finger_body_id or
                               body2 == self.left_finger_body_id or body2 == self.right_finger_body_id)

            if involves_package and involves_finger:
                return True

        return False

    def _get_observation(self):
        obs = []

        # Joint positions (7) - normalized by pi to ~[-1, 1]
        for addr in self.arm_qpos_addrs:
            obs.append(self.data.qpos[addr] / np.pi)

        # Joint velocities (7) - normalized by 5.0
        for addr in self.arm_qvel_addrs:
            obs.append(self.data.qvel[addr] / 5.0)

        # Gripper state (1) - 0 = closed, 1 = open
        finger_pos = self._get_finger_qpos()
        obs.append(finger_pos / 0.04)

        # Gripper holding (1)
        obs.append(1.0 if self.gripper_holding else 0.0)

        # Gripper-to-package vector (3)
        package_pos = self._get_package_position()
        gripper_pos = self._get_gripper_position()
        gripper_to_pkg = package_pos - gripper_pos
        obs.extend(gripper_to_pkg)

        # Package-to-bin vector (3)
        bin_pos = self._get_bin_position()
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
