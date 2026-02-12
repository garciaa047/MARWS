"""MARWS Environment - Franka Emika Panda robot with staged rewards."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class MarwsEnv(gym.Env):
    """Panda robot arm picking a package and placing it on a platform.

    Uses robosuite-style staged rewards:
        - Reaching: 0.0 - 0.25 (distance + orientation + pre-grasp + descent + gripper close)
        - Contact: 0.25 - 0.35 (finger-package contact while gripper closing)
        - Grasping: 0.0 or 0.40 (holding package properly)
        - Lifting:  0.40 - 0.55 (lift height)
        - Hovering: 0.55 - 0.75 (XY distance to bin + height maintenance)
        - Placing:  1.0 (package in bin)

    Reach reward components:
        - Distance to package (0-0.05)
        - Gripper pointing downward (0-0.05)
        - Pre-grasp: above package with XY alignment (0-0.05)
        - Descent bonus: reward for lowering to grasp height when aligned (0-0.05)
        - Gripper closing: reward for closing gripper when positioned to grasp (0-0.05)

    Contact bridge: rewards finger-package contact with partial gripper
    closure, smoothly transitioning from reach to grasp.

    Stability: velocity penalty discourages jerky movements.
    Grasp validation: uses hysteresis (strict acquire, lenient maintain)
    with a grace period to prevent flicker-drop penalties.
    Agent receives max(staged_rewards) each step.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

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

    # Table surface height (table pos z=0.50 + half-height 0.02)
    TABLE_SURFACE_Z = 0.52

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Control frequency settings
        self.control_freq = 20  # Policy runs at 20 Hz
        self.sim_freq = 500     # MuJoCo default
        self.sim_steps_per_control = self.sim_freq // self.control_freq  # 25 steps

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("simulation/franka_emika_panda/warehouse_scene.xml")
        self.data = mujoco.MjData(self.model)

        # Cache IDs for faster access
        self._cache_ids()

        # Observation space (23-dim):
        # - joint_positions: 7
        # - joint_velocities: 7
        # - gripper_state: 1
        # - gripper_holding: 1
        # - gripper_to_package_vector: 3
        # - package_to_bin_vector: 3
        # - package_height_above_table: 1
        obs_dim = 7 + 7 + 1 + 1 + 3 + 3 + 1  # = 23

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 7 joint deltas + 1 gripper command (all normalized [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Core state
        self.viewer = None
        self.gripper_holding = False
        self.package_delivered = False
        self.package_fell = False

        # Delta action control
        self.delta_action_scale = 0.05
        self.gripper_delta_scale = 0.15  # 3x faster gripper response
        self.current_target = None

        # Grasp detection - requires sustained contact
        self.grasp_contact_frames = 0
        self.GRASP_CONTACT_THRESHOLD = 3

        # Grasp release grace period (prevents flicker-drop penalties)
        self.frames_since_grasp_lost = 0
        self.GRASP_GRACE_FRAMES = 10  # Allow 10 frames before counting as a drop

        # Reward tracking
        self.prev_gripper_holding = False
        self.initial_package_z = None

    def _cache_ids(self):
        """Cache joint and body IDs for faster runtime access."""
        self.arm_joint_ids = [
            self.model.joint(f"joint{i}").id for i in range(1, 8)
        ]
        self.arm_qpos_addrs = [
            self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids
        ]
        self.arm_qvel_addrs = [
            self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids
        ]

        self.finger_joint_id = self.model.joint("finger_joint1").id
        self.finger_qpos_addr = self.model.jnt_qposadr[self.finger_joint_id]

        self.hand_body_id = self.model.body("hand").id
        self.package_body_id = self.model.body("package_0").id
        self.bin_body_id = self.model.body("bin_A").id

        self.package_geom_id = self.model.geom("package_0_geom").id
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose to Panda's "home" configuration
        home_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        for i, addr in enumerate(self.arm_qpos_addrs):
            self.data.qpos[addr] = home_qpos[i]

        self.data.qpos[self.finger_qpos_addr] = 0.04  # Gripper open

        for i in range(7):
            self.data.ctrl[i] = home_qpos[i]
        self.data.ctrl[7] = 255  # Gripper open

        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self.gripper_holding = False
        self.package_delivered = False
        self.package_fell = False
        self.current_step = 0

        # Reset delta action control
        self.current_target = np.zeros(8)
        self.current_target[7] = 1.0  # Gripper open

        # Reset grasp detection
        self.grasp_contact_frames = 0
        self.frames_since_grasp_lost = 0

        # Reset reward tracking
        self.prev_gripper_holding = False
        self.initial_package_z = self._get_package_position()[2]

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1

        # Delta action control (separate scale for gripper)
        action = np.array(action, dtype=np.float64)
        delta = action * self.delta_action_scale
        delta[7] = action[7] * self.gripper_delta_scale
        self.current_target = np.clip(self.current_target + delta, -1.0, 1.0)

        # Map to joint control ranges
        for i in range(7):
            ctrl_min, ctrl_max = self.JOINT_CTRL_RANGES[i]
            self.data.ctrl[i] = ctrl_min + (self.current_target[i] + 1) / 2 * (ctrl_max - ctrl_min)

        # Gripper control
        self.data.ctrl[7] = (self.current_target[7] + 1) / 2 * 255

        # Step simulation at control frequency
        for _ in range(self.sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # Update gripper holding state
        self._update_gripper_holding()

        # Compute staged reward
        reward = self._compute_reward()

        # Check termination
        terminated = self.package_delivered or self.package_fell
        truncated = self.current_step >= self.max_steps

        # Info for logging
        staged = self.staged_rewards()
        info = {
            "success": self.package_delivered,
            "stage_rewards": staged,
            "highest_stage": self._get_highest_stage(staged),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_package_position(self):
        return self.data.xpos[self.package_body_id].copy()

    def _get_bin_position(self):
        return self.data.xpos[self.bin_body_id].copy()

    def _get_gripper_position(self):
        return self.data.xpos[self.hand_body_id].copy()

    def _get_finger_qpos(self):
        return self.data.qpos[self.finger_qpos_addr]

    def _get_gripper_downward_alignment(self):
        """Get how well the gripper is pointing downward (0 to 1).

        Returns 1.0 when gripper z-axis points straight down,
        0.0 when pointing horizontally or up.
        """
        # Get rotation matrix of hand (3x3 flattened to 9)
        hand_xmat = self.data.xmat[self.hand_body_id].reshape(3, 3)
        # Gripper z-axis in world coordinates (third column)
        gripper_z = hand_xmat[:, 2]
        # How much does it point down? (dot product with -z world axis)
        downward = -gripper_z[2]  # Negative because we want pointing DOWN
        # Clamp to [0, 1]
        return max(0.0, downward)

    def staged_rewards(self):
        """Calculate rewards for each stage.

        Returns:
            tuple: (reach, contact, grasp, lift, hover, place) rewards
        """
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        # Reaching: 0 to 0.25 (distance + orientation + position + gripper close)
        # Distance component: 0 to 0.05
        dist_to_package = np.linalg.norm(gripper_pos - package_pos)
        dist_reward = (1 - np.tanh(5.0 * dist_to_package)) * 0.05

        # Orientation component: 0 to 0.05 (gripper pointing down)
        downward = self._get_gripper_downward_alignment()
        orient_reward = downward * 0.05

        # Pre-grasp position: 0 to 0.15 (XY alignment + descent + gripper closing)
        xy_dist = np.linalg.norm(gripper_pos[:2] - package_pos[:2])
        height_above = gripper_pos[2] - package_pos[2]
        above_package = height_above > 0  # Gripper higher than package
        pregrasp_reward = 0.0
        if above_package and downward > 0.5:  # Must be pointing down
            # Reward for XY alignment (closer = better) - up to 0.05
            # Wider gradient (tanh 5.0) so agent sees signal at larger distances
            xy_alignment = (1 - np.tanh(5.0 * xy_dist)) * 0.05
            pregrasp_reward = xy_alignment

            # Descent bonus: reward getting closer to grasping height when XY-aligned
            # Relaxed XY threshold (8cm) and gentler tanh (5.0) for wider gradient
            if xy_dist < 0.08:
                # Target height is ~3cm above package (grasping position)
                target_height = 0.03
                height_error = abs(height_above - target_height)
                descent_bonus = (1 - np.tanh(5.0 * height_error)) * 0.05
                pregrasp_reward += descent_bonus

                # Gripper closing bonus: 0 to 0.05
                # Relaxed height threshold (12cm) so agent learns to close
                # earlier in the approach rather than only at the last moment
                if height_above < 0.12:
                    finger_pos = self._get_finger_qpos()
                    # finger_pos: 0.04 = fully open, 0 = fully closed
                    gripper_closedness = 1.0 - (finger_pos / 0.04)
                    pregrasp_reward += gripper_closedness * 0.05

        # Combined reach reward (max 0.25)
        r_reach = dist_reward + orient_reward + pregrasp_reward

        # Contact bridge: 0.25 to 0.35 (finger touching package + gripper closing)
        # Bridges the gap between reach (0.25 max) and grasp (0.40).
        # The agent gets rewarded for making contact and closing, even before
        # the full grasp criteria (sustained contact for N frames) are met.
        r_contact = 0.0
        if self._check_finger_package_contact() and above_package:
            finger_pos = self._get_finger_qpos()
            gripper_closedness = 1.0 - (finger_pos / 0.04)
            if gripper_closedness > 0.2:  # At least starting to close
                r_contact = 0.25 + min(gripper_closedness, 1.0) * 0.10

        # Grasping: 0 or 0.40
        # Must be a valid grasp: gripper above package (within XY tolerance)
        r_grasp = 0.0
        if self.gripper_holding:
            xy_dist_grasp = np.linalg.norm(gripper_pos[:2] - package_pos[:2])
            gripper_above = gripper_pos[2] > package_pos[2] - 0.05  # Allow slight tolerance
            if xy_dist_grasp < 0.08 and gripper_above:
                r_grasp = 0.40

        # Lifting: 0.40 to 0.55 (only if valid grasp)
        r_lift = 0.0
        lift_height = package_pos[2] - self.initial_package_z
        if r_grasp > 0:  # Only count lift if grasp is valid
            # Also verify package is still near gripper (not knocked away)
            if dist_to_package < 0.15:
                target_height = 0.15  # 15cm target lift
                height_progress = max(0, min(lift_height / target_height, 1.0))
                r_lift = 0.40 + height_progress * 0.15

        # Hovering: 0.55 to 0.75 (only if lifted enough with valid grasp)
        # Includes height maintenance: agent must keep the package above the
        # table surface during transport to avoid dragging it into the edge.
        r_hover = 0.0
        if r_lift > 0.50 and lift_height > 0.08:  # Must have actually lifted
            dist_to_bin = np.linalg.norm(package_pos[:2] - bin_pos[:2])
            # XY approach reward: 0 to 0.15
            xy_approach = (1 - np.tanh(5.0 * dist_to_bin)) * 0.15
            # Height maintenance reward: 0 to 0.05
            # Reward keeping package well above the table surface (>=10cm)
            clearance = package_pos[2] - self.TABLE_SURFACE_Z
            height_bonus = min(max(clearance / 0.10, 0.0), 1.0) * 0.05
            r_hover = 0.55 + xy_approach + height_bonus

        # Placing: 1.0 (package in bin)
        r_place = 1.0 if self._check_package_in_bin() else 0.0

        return r_reach, r_contact, r_grasp, r_lift, r_hover, r_place

    def _get_highest_stage(self, staged):
        """Return name of highest achieved stage."""
        r_reach, r_contact, r_grasp, r_lift, r_hover, r_place = staged
        if r_place > 0:
            return "place"
        elif r_hover > 0:
            return "hover"
        elif r_lift > 0.40:
            return "lift"
        elif r_grasp > 0:
            return "grasp"
        elif r_contact > 0:
            return "contact"
        else:
            return "reach"

    def _compute_reward(self):
        """Main reward function using staged rewards."""
        staged = self.staged_rewards()
        reward = max(staged)

        # Stability penalty: penalize high joint velocities (reduces jerkiness)
        # This encourages smooth, controlled movements
        joint_velocities = np.array([self.data.qvel[addr] for addr in self.arm_qvel_addrs])
        velocity_magnitude = np.linalg.norm(joint_velocities)
        # Small penalty scaled by velocity (max ~0.02 penalty at high velocities)
        stability_penalty = 0.005 * min(velocity_magnitude, 4.0)
        reward -= stability_penalty

        # Penalty for dropping package (not over bin).
        # Only penalize when the package has truly fallen away from
        # the gripper, not during momentary contact flicker (the grace
        # period in _update_gripper_holding handles debouncing).
        if self.prev_gripper_holding and not self.gripper_holding:
            package_pos = self._get_package_position()
            gripper_pos = self._get_gripper_position()
            drop_dist = np.linalg.norm(gripper_pos - package_pos)
            # Only penalize if package has actually separated AND is not in the bin
            if drop_dist > 0.10 and not self._check_package_in_bin():
                reward -= 0.5

        # Penalty for package falling off table
        package_pos = self._get_package_position()
        if package_pos[2] < 0:
            reward -= 1.0
            self.package_fell = True

        # Success!
        if self._check_package_in_bin():
            self.package_delivered = True

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
        """Update gripper holding state based on contact detection.

        Uses hysteresis: stricter conditions to acquire grasp, more
        lenient conditions to maintain it during transport. Also tracks
        a grace period so momentary contact loss doesn't instantly
        count as a drop.
        """
        finger_touching = self._check_finger_package_contact()
        finger_pos = self._get_finger_qpos()
        gripper_closed_enough = finger_pos < 0.025

        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()
        dist_to_package = np.linalg.norm(gripper_pos - package_pos)

        if self.gripper_holding:
            # Maintaining grasp: more lenient - just check the package
            # is still close to the gripper and fingers are closed.
            # During transport the package can shift slightly.
            still_holding = (
                finger_touching
                and gripper_closed_enough
                and dist_to_package < 0.12
            )
            if still_holding:
                self.frames_since_grasp_lost = 0
            else:
                self.frames_since_grasp_lost += 1
                # Grace period: don't drop immediately on brief contact loss
                if self.frames_since_grasp_lost > self.GRASP_GRACE_FRAMES:
                    self.gripper_holding = False
                    self.grasp_contact_frames = 0
                    self.frames_since_grasp_lost = 0
        else:
            # Acquiring grasp: stricter positioning requirements
            xy_dist = np.linalg.norm(gripper_pos[:2] - package_pos[:2])
            gripper_above = gripper_pos[2] > package_pos[2] - 0.02
            properly_positioned = xy_dist < 0.06 and gripper_above

            if finger_touching and gripper_closed_enough and properly_positioned:
                self.grasp_contact_frames += 1
                if self.grasp_contact_frames >= self.GRASP_CONTACT_THRESHOLD:
                    self.gripper_holding = True
                    self.frames_since_grasp_lost = 0
            else:
                self.grasp_contact_frames = 0

    def _check_finger_package_contact(self):
        """Check if either finger is in contact with the package."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            involves_package = (geom1 == self.package_geom_id or geom2 == self.package_geom_id)
            involves_finger = (body1 == self.left_finger_body_id or body1 == self.right_finger_body_id or
                               body2 == self.left_finger_body_id or body2 == self.right_finger_body_id)

            if involves_package and involves_finger:
                return True

        return False

    def _get_observation(self):
        obs = []

        # Joint positions (7)
        for addr in self.arm_qpos_addrs:
            obs.append(self.data.qpos[addr] / np.pi)

        # Joint velocities (7)
        for addr in self.arm_qvel_addrs:
            obs.append(self.data.qvel[addr] / 5.0)

        # Gripper state (1)
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

        # Package height above table surface (1) - helps agent learn clearance
        height_above_table = package_pos[2] - self.TABLE_SURFACE_Z
        obs.append(height_above_table)

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
