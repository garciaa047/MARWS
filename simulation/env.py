"""MARWS Environment - Single robot arm pick-and-place task."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class MarwsEnv(gym.Env):
    """Single robot arm picking a package and placing it in a bin."""

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
        # - gripper_to_package_vector: 3 (critical for precise grasping)
        obs_dim = 6 + 6 + 1 + 1 + 3 + 3 + 3  # = 23

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 6 joint velocities + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Internal state
        self.viewer = None
        self.gripper_holding = False
        self.package_delivered = False

        # Reward shaping state
        self.prev_gripper_holding = False
        self.prev_gripper_pos = None
        self.prev_dist_to_target = None
        self.reached_package = False  # One-time milestone
        self.made_contact = False  # One-time milestone
        self.grasped_once = False  # One-time milestone for grasp bonus

        # Contact persistence for stable grasp detection
        self.contact_frames = 0  # Count consecutive frames with contact
        self.no_contact_frames = 0  # Count consecutive frames without contact
        self.CONTACT_THRESHOLD = 2  # Frames needed to confirm grasp (lowered)
        self.RELEASE_THRESHOLD = 8  # Frames needed to confirm release (increased)

        # Additional milestones for better reward shaping
        self.enclosed_package = False  # One-time: package between fingers
        self.closed_on_package = False  # One-time: gripper closed while enclosed
        self.lifted_package = False  # One-time: package lifted off table
        self.near_bin = False  # One-time: package brought near bin
        self.initial_package_z = 0.175  # Track initial package height (default)
        self.initial_package_xy = None  # Track initial package XY position
        self.prev_package_pos = None  # Track package movement for push detection

        # Anti-hovering: track how long gripper has been open while positioned
        self.hover_frames = 0

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

        self.gripper_holding = False
        self.package_delivered = False
        self.current_step = 0

        # Reset reward shaping state
        self.prev_gripper_holding = False
        self.prev_gripper_pos = None
        self.prev_dist_to_target = None
        self.reached_package = False
        self.made_contact = False
        self.grasped_once = False

        # Reset contact persistence
        self.contact_frames = 0
        self.no_contact_frames = 0

        # Reset additional milestones
        self.enclosed_package = False
        self.closed_on_package = False
        self.lifted_package = False
        self.near_bin = False
        package_pos = self._get_package_position()
        self.initial_package_z = package_pos[2]
        self.initial_package_xy = package_pos[:2].copy()
        self.prev_package_pos = package_pos.copy()

        # Reset anti-hovering counter
        self.hover_frames = 0

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1

        # Apply actions to actuators
        max_vel = 2.0
        self.data.ctrl[:6] = action[:6] * max_vel

        # Gripper control (single actuator, equality constraint moves both fingers)
        gripper_cmd = (action[6] + 1) / 2 * 0.04
        self.data.ctrl[6] = gripper_cmd

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Update gripper holding state
        self._update_gripper_holding()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self.package_delivered
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
        """
        Progressive milestone-based reward structure:
        - Distance shaping to guide toward package/bin
        - Graduated milestones: reach → enclosed → contact → grasp → lift → deliver
        - Intermediate rewards to bridge the gap between milestones
        """
        reward = 0.0
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        if self.package_delivered:
            return 0.0

        dist_to_package = np.linalg.norm(gripper_pos - package_pos)
        dist_to_bin = np.linalg.norm(package_pos - bin_pos)

        # Check for contacts and enclosure
        contacts = self._get_gripper_contacts()
        has_contact = len(contacts) > 0
        is_enclosed = self._is_package_enclosed()

        # Gripper closure amount (0 = open, 1 = closed)
        gripper_closed_amt = 1.0 - (self.data.qpos[6] / 0.04)

        if not self.gripper_holding:
            # === PHASE 1: Approach and grasp package ===

            # Distance shaping (guides toward package)
            if self.prev_gripper_pos is not None:
                prev_dist = np.linalg.norm(self.prev_gripper_pos - package_pos)
                improvement = prev_dist - dist_to_package
                reward += 2.0 * improvement

            # Bonus for good posture: above package, not too far, gripper ready
            if gripper_pos[2] > package_pos[2] + 0.02 and dist_to_package < 0.15:
                reward += 0.05  # Bonus for good positioning
                # Extra bonus if gripper is at least partially closed (ready to grasp)
                if gripper_closed_amt > 0.3:
                    reward += 0.05

            # MILESTONE 1: Reached package vicinity (one-time, within 5cm)
            # Only triggers if approaching from above
            if dist_to_package < 0.05 and not self.reached_package:
                if gripper_pos[2] > package_pos[2]:  # Must be above package
                    reward += 5.0
                    self.reached_package = True

            # Early gripper closing reward - only when close AND approaching from above
            # This encourages top-down approach with gripper ready to close
            if dist_to_package < 0.08 and gripper_pos[2] > package_pos[2]:
                reward += 0.3 * gripper_closed_amt  # Increased from 0.15
                # Penalty for approaching with wide open gripper
                if gripper_closed_amt < 0.3:
                    reward -= 0.1

            # MILESTONE 2: Package enclosed by gripper (one-time)
            if is_enclosed and not self.enclosed_package:
                reward += 10.0
                self.enclosed_package = True

            # MILESTONE 3: Made contact with package (one-time)
            if has_contact and not self.made_contact:
                reward += 10.0
                self.made_contact = True

            # Check if gripper is properly above the package (not pushing into it)
            gripper_above_package = gripper_pos[2] > package_pos[2] + 0.02

            # Intermediate shaping: reward closing gripper when package is enclosed
            # Only if package is still in place (not pushed) AND gripper is above
            if is_enclosed and gripper_above_package:
                # Strong reward for closing when properly positioned
                reward += 1.0 * gripper_closed_amt  # Increased from 0.6

                # MODERATE PENALTY for hovering with open gripper
                # Reduced from escalating to fixed to avoid extreme avoidance behavior
                if gripper_closed_amt < 0.5:  # Gripper mostly open
                    self.hover_frames += 1
                    # Fixed moderate penalty instead of escalating
                    reward -= 0.3  # Constant penalty for open gripper while positioned
                else:
                    self.hover_frames = max(0, self.hover_frames - 2)  # Reduce if closing

                # PENALTY for keeping gripper open when properly positioned
                gripper_open_amt = 1.0 - gripper_closed_amt
                reward -= 0.3 * gripper_open_amt  # Increased from 0.2

                # MILESTONE: Closed gripper on package (one-time, bridges to grasp)
                if gripper_closed_amt > 0.7 and not self.closed_on_package:
                    reward += 20.0  # Increased from 15
                    self.closed_on_package = True

            # Extra shaping: reward closing gripper when in contact AND above package
            if has_contact and gripper_above_package:
                reward += 0.8 * gripper_closed_amt  # Increased from 0.4

        else:
            # === PHASE 2: Carry to bin ===

            # MILESTONE 4: Lifted package (one-time) - requires clear lift
            if not self.lifted_package:
                if package_pos[2] > self.initial_package_z + 0.05:  # 5cm lift required
                    reward += 20.0
                    self.lifted_package = True

            # MILESTONE 5: Near bin (one-time, within 15cm)
            if not self.near_bin and dist_to_bin < 0.15:
                reward += 20.0
                self.near_bin = True

            # Distance shaping toward bin (stronger incentive)
            if self.prev_dist_to_target is not None:
                improvement = self.prev_dist_to_target - dist_to_bin
                reward += 5.0 * improvement  # Increased from 3.0

            if self.prev_dist_to_target is None:
                self.prev_dist_to_target = dist_to_bin

            self.prev_dist_to_target = dist_to_bin

            # Per-step reward for maintaining hold (encourages faster delivery)
            reward += 0.03

        # === ONE-TIME MILESTONE BONUSES ===

        # MILESTONE 5: Successful grasp (truly one-time) - BIG reward to encourage grasping
        if self.gripper_holding and not self.grasped_once:
            reward += 50.0  # Increased from 30 to strongly encourage grasping
            self.grasped_once = True
            self.prev_dist_to_target = dist_to_bin
            self.hover_frames = 0  # Reset hover counter on successful grasp

        # MILESTONE 6: Delivery (one-time)
        if self._check_package_in_bin():
            reward += 100.0
            self.package_delivered = True

        # === PENALTIES ===

        # Dropped package after grasping
        if self.prev_gripper_holding and not self.gripper_holding:
            if not self.package_delivered:
                reward -= 5.0

        # PENALTY: Pushing package without holding it
        # Detects horizontal movement when not grasping - discourages ramming
        if self.prev_package_pos is not None and not self.gripper_holding:
            package_xy_movement = np.linalg.norm(package_pos[:2] - self.prev_package_pos[:2])
            if package_xy_movement > 0.005:  # Moved more than 5mm (relaxed from 3mm)
                reward -= 2.0 * package_xy_movement  # Reduced from 3.0

            # PENALTY: Pushing package DOWN into the table
            package_z_drop = self.prev_package_pos[2] - package_pos[2]
            if package_z_drop > 0.003:  # Relaxed from 0.002
                reward -= 3.0 * package_z_drop  # Reduced from 5.0

        # Package fell off table
        if package_pos[2] < 0:
            reward -= 10.0
            self.package_delivered = True  # End episode

        # Package pushed too far from initial position (likely knocked off table area)
        package_xy_drift = np.linalg.norm(package_pos[:2] - self.initial_package_xy)
        if package_xy_drift > 0.3 and not self.gripper_holding:
            reward -= 5.0

        # PENALTY: Robot arm/gripper touching the table
        table_contacts = self._get_robot_table_contacts()
        if table_contacts > 0:
            reward -= 0.5 * table_contacts  # Penalty per contact point

        # Update state
        self.prev_gripper_pos = gripper_pos.copy()
        self.prev_gripper_holding = self.gripper_holding
        self.prev_package_pos = package_pos.copy()
        if not self.gripper_holding:
            self.prev_dist_to_target = None

        return float(reward)

    def _check_package_in_bin(self):
        package_pos = self._get_package_position()
        bin_pos = self._get_bin_position()

        dx = abs(package_pos[0] - bin_pos[0])
        dy = abs(package_pos[1] - bin_pos[1])
        dz = package_pos[2] - bin_pos[2]

        return dx < 0.12 and dy < 0.12 and 0 < dz < 0.25

    def _update_gripper_holding(self):
        """Update gripper holding - requires package to actually follow gripper (be lifted)."""
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()

        # Check if package is within gripper grasp zone
        dx = abs(package_pos[0] - gripper_pos[0])
        dy = abs(package_pos[1] - gripper_pos[1])
        dz = package_pos[2] - gripper_pos[2]

        # Package must be very close to gripper (tighter tolerance)
        package_in_grasp_zone = dx < 0.02 and dy < 0.02 and -0.04 < dz < 0.01

        # Gripper must be closed enough to hold the 3cm package
        gripper_closed_enough = self.data.qpos[6] < 0.008

        # CRITICAL: Package must be lifted off the table to confirm actual grasp
        # This prevents "pushing" from counting as grasping
        package_lifted = package_pos[2] > self.initial_package_z + 0.01

        # Grasp condition: in zone AND closed AND lifted (or already holding)
        is_grasping = package_in_grasp_zone and gripper_closed_enough and (package_lifted or self.gripper_holding)

        if is_grasping:
            self.contact_frames += 1
            self.no_contact_frames = 0
        else:
            self.no_contact_frames += 1
            self.contact_frames = 0

        # Hysteresis for stability
        if not self.gripper_holding:
            if self.contact_frames >= self.CONTACT_THRESHOLD:
                self.gripper_holding = True
        else:
            if self.no_contact_frames >= self.RELEASE_THRESHOLD:
                self.gripper_holding = False

    def _get_gripper_contacts(self):
        contacts = []
        finger_geom_ids = [
            self.model.geom("finger_left_geom").id,
            self.model.geom("finger_right_geom").id
        ]
        pkg_geom_id = self.model.geom("package_0_geom").id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            for finger_id in finger_geom_ids:
                if (geom1 == finger_id and geom2 == pkg_geom_id) or \
                   (geom1 == pkg_geom_id and geom2 == finger_id):
                    contacts.append(0)
        return contacts

    def _is_package_enclosed(self):
        """Check if package is positioned between the gripper fingers (stricter check)."""
        gripper_pos = self._get_gripper_position()
        package_pos = self._get_package_position()

        # Get relative position in gripper frame
        dx = abs(package_pos[0] - gripper_pos[0])
        dy = abs(package_pos[1] - gripper_pos[1])
        dz = package_pos[2] - gripper_pos[2]

        # Package is "enclosed" if it's well centered under the gripper
        # Tighter tolerances to prevent rewarding just being nearby
        # Also require package hasn't been pushed far from start
        package_xy_drift = np.linalg.norm(package_pos[:2] - self.initial_package_xy)
        package_stable = package_xy_drift < 0.1  # Package hasn't been pushed around

        return dx < 0.025 and dy < 0.025 and -0.05 < dz < 0.01 and package_stable

    def _get_robot_table_contacts(self):
        """Check if robot arm or gripper is touching the table."""
        contacts = 0
        table_geom_id = self.model.geom("table_top").id

        # Robot geoms that shouldn't touch the table
        robot_geom_names = [
            "gripper_base_geom", "finger_left_geom", "finger_right_geom",
            "link4_geom", "link5_geom", "link6_geom"
        ]
        robot_geom_ids = [self.model.geom(name).id for name in robot_geom_names]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            for robot_id in robot_geom_ids:
                if (geom1 == robot_id and geom2 == table_geom_id) or \
                   (geom1 == table_geom_id and geom2 == robot_id):
                    contacts += 1
        return contacts

    def _get_observation(self):
        obs = []

        # Joint positions
        obs.extend(self.data.qpos[:6])

        # Joint velocities
        obs.extend(self.data.qvel[:6])

        # Gripper state (single value, both fingers coupled)
        gripper_pos = self.data.qpos[6]
        obs.append(gripper_pos / 0.04)

        # Gripper holding
        obs.append(1.0 if self.gripper_holding else 0.0)

        # Package position
        package_pos = self._get_package_position()
        obs.extend(package_pos)

        # Bin position
        obs.extend(self._get_bin_position())

        # Gripper-to-package vector (critical for learning precise positioning)
        gripper_pos_world = self._get_gripper_position()
        gripper_to_pkg = package_pos - gripper_pos_world
        obs.extend(gripper_to_pkg)

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
