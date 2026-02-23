"""MARWS Goal-Conditioned Environment -- Panda pick-and-place for HER.

Architecture modeled on gymnasium_robotics FetchPickAndPlace:
- Dict observation: {observation, achieved_goal, desired_goal}
- Externalized compute_reward(achieved_goal, desired_goal, info)
- Dense reward: -||achieved - desired|| (negative Euclidean distance)
- Cartesian action space (4-dim) via Jacobian IK
- Short episodes (100 steps) with no early termination
- Randomized object position and goal sampling

Designed for SAC + HER (Hindsight Experience Replay).
"""
import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


def goal_distance(goal_a, goal_b):
    """Euclidean distance between goals. Supports batched inputs."""
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


MODEL_XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "franka_emika_panda",
    "warehouse_scene.xml",
)


class PandaPickAndPlaceEnv(gym.Env):
    """Franka Emika Panda pick-and-place with goal-conditioned dense reward.

    Observation (Dict):
        observation (25-dim): TCP pos, object pos, relative pos, gripper state,
                              object rotation, object velocity, TCP velocity,
                              gripper velocity
        achieved_goal (3-dim): current object position [x, y, z]
        desired_goal  (3-dim): target position [x, y, z]

    Action (4-dim, continuous [-1, 1]):
        [0:3] Cartesian displacement of TCP (dx, dy, dz), scaled by 0.05m
        [3]   Gripper command: +1 = open, -1 = close

    Reward:
        dense:  -||achieved - desired|| (default, matches Fetch)
        sparse: 0 if ||achieved - desired|| < threshold, else -1

    Episode ends only by truncation (100 steps, no early termination).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    # ---- Constants ----
    DISTANCE_THRESHOLD = 0.05      # 5cm success radius
    MIN_GOAL_DISTANCE = 0.08       # 8cm min distance between object start and goal

    # Approach shaping constants (step() only, not compute_reward)
    XY_APPROACH_RADIUS = 0.02      # XY penalty saturates within 2cm of object center
    XY_APPROACH_WEIGHT = 1.0       # Weight of horizontal alignment reward
    DESCENT_XY_THRESHOLD = 0.05    # Z descent reward only fires when XY within 5cm
    Z_APPROACH_RADIUS = 0.01       # Z penalty saturates within 1cm of object height
    Z_APPROACH_WEIGHT = 1.5        # Weight of Z descent reward
    GRASP_ZONE = 0.05              # Lift shaping active when TCP within 5cm of object
    W_LIFT = 5.0                   # Lift shaping weight (reward per meter of object rise)
    CARTESIAN_SCALE = 0.05         # Max 5cm displacement per step
    N_SUBSTEPS = 10                # Sim substeps per control step (20ms at 0.002s timestep)
    MAX_EPISODE_STEPS = 100        # ~4 seconds at 25 Hz

    # Panda joint limits (7 revolute)
    JOINT_LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                                  -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                                   2.8973, 3.7525, 2.8973])

    # TCP offset from hand body center, in hand-local frame.
    # Fingers attach at z=0.0584, fingertip pad center at z~0.0445 from finger.
    TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.103])

    # Object randomization (XY jitter on table)
    OBJ_RANGE = 0.06  # +/- 6cm (was 3cm; more diversity in initial conditions)

    # TCP workspace bounds — prevents arm from going below table / out of reach.
    # Without these, the agent learns degenerate policies (e.g. driving TCP into floor).
    # Table surface at z=0.52, package center at z=0.54.
    TCP_BOUNDS_LOW = np.array([0.25, -0.55, 0.42])
    TCP_BOUNDS_HIGH = np.array([0.75, 0.55, 0.80])

    # Damping for pseudoinverse IK (prevents singularity blow-up)
    IK_DAMPING = 1e-4
    # Max joint delta per step (radians) - safety clamp
    MAX_DQ = 0.5

    def __init__(self, render_mode=None, reward_type="dense",
                 randomize_object=False, randomize_goals=True):
        super().__init__()
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.randomize_object = randomize_object
        self.randomize_goals = randomize_goals

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Cache IDs for fast runtime access
        self._cache_ids()

        # Set up initial state (home configuration)
        self._setup_initial_state()

        # Previous object position for lift shaping (delta_z)
        self._prev_obj_pos = self.data.xpos[self.package_body_id].copy()

        # Build observation and action spaces
        self.goal = self._sample_goal()
        obs = self._get_obs()

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64
            ),
            "achieved_goal": spaces.Box(
                -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64
            ),
            "desired_goal": spaces.Box(
                -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float64
            ),
        })

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.current_step = 0
        self.viewer = None

    # ==================================================================
    # Initialization helpers
    # ==================================================================

    def _cache_ids(self):
        """Cache MuJoCo body, joint, and site IDs for fast access."""
        # Arm joints (7 revolute)
        self.arm_joint_ids = [
            self.model.joint(f"joint{i}").id for i in range(1, 8)
        ]
        self.arm_qpos_addrs = np.array([
            self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids
        ])
        self.arm_dof_addrs = np.array([
            self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids
        ])

        # Finger joints (2 slide)
        self.finger_joint1_id = self.model.joint("finger_joint1").id
        self.finger_joint2_id = self.model.joint("finger_joint2").id
        self.finger_qpos1_addr = self.model.jnt_qposadr[self.finger_joint1_id]
        self.finger_qpos2_addr = self.model.jnt_qposadr[self.finger_joint2_id]
        self.finger_dof1_addr = self.model.jnt_dofadr[self.finger_joint1_id]
        self.finger_dof2_addr = self.model.jnt_dofadr[self.finger_joint2_id]

        # Bodies
        self.hand_body_id = self.model.body("hand").id
        self.package_body_id = self.model.body("package_0").id

        # Package freejoint
        self.pkg_joint_id = self.model.joint("package_0_joint").id
        self.pkg_qpos_addr = self.model.jnt_qposadr[self.pkg_joint_id]
        self.pkg_dof_addr = self.model.jnt_dofadr[self.pkg_joint_id]

        # Target visualization site
        try:
            self.target_site_id = self.model.site("target0").id
        except Exception:
            self.target_site_id = None

    def _setup_initial_state(self):
        """Set Panda to home configuration and record initial state."""
        home_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]

        for i, addr in enumerate(self.arm_qpos_addrs):
            self.data.qpos[addr] = home_qpos[i]
        self.data.qpos[self.finger_qpos1_addr] = 0.04
        self.data.qpos[self.finger_qpos2_addr] = 0.04

        for i in range(7):
            self.data.ctrl[i] = home_qpos[i]
        self.data.ctrl[7] = 255  # gripper open

        mujoco.mj_forward(self.model, self.data)

        # Snapshot for fast reset
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()
        self._init_ctrl = self.data.ctrl.copy()

        # Record reference positions
        self.initial_tcp_xpos = self._get_tcp_pos().copy()
        self.height_offset = self.data.xpos[self.package_body_id][2]

    # ==================================================================
    # Core Gymnasium API
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Restore initial state
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos.copy()
        self.data.qvel[:] = self._init_qvel.copy()
        self.data.ctrl[:] = self._init_ctrl.copy()

        # Randomize object XY position on table (disabled by default for
        # focused learning — enable with randomize_object=True once the
        # agent has learned basic skills from a fixed starting position)
        if self.randomize_object:
            obj_xy = self._init_qpos[self.pkg_qpos_addr: self.pkg_qpos_addr + 2].copy()
            obj_xy += self.np_random.uniform(-self.OBJ_RANGE, self.OBJ_RANGE, size=2)
            self.data.qpos[self.pkg_qpos_addr: self.pkg_qpos_addr + 2] = obj_xy

        mujoco.mj_forward(self.model, self.data)

        # Let the package settle onto the table under gravity
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Re-record settled object height
        self.height_offset = self.data.xpos[self.package_body_id][2]

        # Initialize previous object position for lift shaping
        self._prev_obj_pos = self.data.xpos[self.package_body_id].copy()

        # Sample a new goal and update visualization
        self.goal = self._sample_goal()
        self._update_target_viz()

        self.current_step = 0
        obs = self._get_obs()
        return obs, {"is_success": 0.0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.current_step += 1

        # Apply Cartesian control + gripper command
        self._apply_action(action)

        # Step physics
        mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)

        obs = self._get_obs()
        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        tcp_pos = obs["observation"][:3]

        d = goal_distance(ag, dg)
        grip_obj_dist = float(np.linalg.norm(tcp_pos - ag))
        info = {
            "is_success": float(d < self.DISTANCE_THRESHOLD),
            "distance": float(d),
            "object_z": float(ag[2]),
            "goal_z": float(dg[2]),
            "grip_obj_dist": grip_obj_dist,
        }

        # Goal-distance reward (HER-compatible, used for relabeled transitions)
        reward = float(self.compute_reward(ag, dg, info))

        # ---- Approach shaping (step() only, NOT compute_reward) ----
        # Stage 1: XY alignment — pull gripper horizontally over object.
        # Saturates within XY_APPROACH_RADIUS (2cm): strong gradient to approach,
        # no penalty once aligned. Gated stages force top-down grasp geometry.
        xy_dist = float(np.linalg.norm(tcp_pos[:2] - ag[:2]))
        z_diff  = float(tcp_pos[2] - ag[2])   # positive = gripper above object

        xy_penalty = -max(xy_dist - self.XY_APPROACH_RADIUS, 0.0)
        reward += self.XY_APPROACH_WEIGHT * xy_penalty

        # Stage 2: Z descent — descend once XY-aligned (forces top-down approach).
        # Only fires within DESCENT_XY_THRESHOLD (5cm) so the arm is above the
        # object before descending rather than approaching from the side.
        if xy_dist < self.DESCENT_XY_THRESHOLD:
            z_penalty = -max(abs(z_diff) - self.Z_APPROACH_RADIUS, 0.0)
            reward += self.Z_APPROACH_WEIGHT * z_penalty

        # ---- Lift shaping (step() only, NOT compute_reward) ----
        # Fires only when TCP is in grasp zone AND the object is actually rising.
        # Zero for hovering (delta_z=0). Replaces the old grasp closure bonus
        # which created a hover local optimum: the agent must actually move the
        # object upward to earn this reward, not merely close fingers near it.
        delta_z = float(ag[2] - self._prev_obj_pos[2])
        if grip_obj_dist < self.GRASP_ZONE and delta_z > 0:
            reward += self.W_LIFT * delta_z

        # Update previous object position for lift shaping
        self._prev_obj_pos = ag.copy()

        terminated = False  # continuing task, no early termination
        truncated = self.current_step >= self.MAX_EPISODE_STEPS

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # GoalEnv interface (required by SB3 HerReplayBuffer)
    # ------------------------------------------------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Vectorized reward computation for HER goal relabeling.

        Args:
            achieved_goal: (..., 3) object position(s)
            desired_goal:  (..., 3) target position(s)
            info: unused (HER compatibility)

        Returns:
            reward: (...,) float array
        """
        d = goal_distance(np.asarray(achieved_goal), np.asarray(desired_goal))
        if self.reward_type == "sparse":
            return -(d > self.DISTANCE_THRESHOLD).astype(np.float32)
        else:
            return -d.astype(np.float32)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return False

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False

    # ==================================================================
    # Action handling -- Cartesian IK control
    # ==================================================================

    def _apply_action(self, action):
        """Convert 4-dim Cartesian action to joint-space control commands.

        action[0:3]: Cartesian displacement (dx, dy, dz) in [-1, 1]
        action[3]:   Gripper command (+1 = open, -1 = close)
        """
        pos_delta = action[:3] * self.CARTESIAN_SCALE

        # Clip desired TCP position to workspace bounds
        tcp_pos = self._get_tcp_pos()
        target_tcp = np.clip(
            tcp_pos + pos_delta, self.TCP_BOUNDS_LOW, self.TCP_BOUNDS_HIGH
        )
        pos_delta = target_tcp - tcp_pos

        # Compute Jacobian at TCP for the arm joints
        jacp_full = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model, self.data, jacp_full, None,
            np.ascontiguousarray(tcp_pos), self.hand_body_id,
        )
        J = jacp_full[:, self.arm_dof_addrs]  # (3, 7)

        # Damped pseudoinverse: dq = J^T (J J^T + lambda^2 I)^{-1} dx
        JJT = J @ J.T + self.IK_DAMPING * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, pos_delta)

        # Safety clamp on joint deltas
        dq = np.clip(dq, -self.MAX_DQ, self.MAX_DQ)

        # Compute new joint targets
        current_q = self.data.qpos[self.arm_qpos_addrs].copy()
        target_q = np.clip(
            current_q + dq, self.JOINT_LIMITS_LOW, self.JOINT_LIMITS_HIGH
        )

        # Set arm actuator targets (PD controller tracks these)
        for i in range(7):
            self.data.ctrl[i] = target_q[i]

        # Gripper: map [-1, +1] -> [0, 255]
        # +1 = open (ctrl=255), -1 = close (ctrl=0)
        self.data.ctrl[7] = np.clip((action[3] + 1.0) / 2.0 * 255.0, 0.0, 255.0)

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_obs(self):
        """Build goal-conditioned observation dict.

        observation (25-dim):
            tcp_pos          (3)  - TCP world position
            object_pos       (3)  - object world position
            object_rel_pos   (3)  - object - TCP
            gripper_state    (2)  - left/right finger joint positions
            object_rot       (3)  - object Euler angles
            object_velp      (3)  - object linear velocity (relative, * dt)
            object_velr      (3)  - object angular velocity (* dt)
            tcp_velp         (3)  - TCP linear velocity (* dt)
            gripper_vel      (2)  - finger joint velocities (* dt)
        """
        dt = self.model.opt.timestep * self.N_SUBSTEPS

        tcp_pos = self._get_tcp_pos()
        tcp_vel = self._get_tcp_vel() * dt

        object_pos = self.data.xpos[self.package_body_id].copy()
        object_rel_pos = object_pos - tcp_pos

        # Object rotation (quaternion -> Euler)
        obj_quat = self.data.qpos[
            self.pkg_qpos_addr + 3: self.pkg_qpos_addr + 7
        ].copy()
        object_rot = _quat_to_euler(obj_quat)

        # Object velocity (from freejoint DOFs)
        object_velp = self.data.qvel[
            self.pkg_dof_addr: self.pkg_dof_addr + 3
        ].copy() * dt
        object_velr = self.data.qvel[
            self.pkg_dof_addr + 3: self.pkg_dof_addr + 6
        ].copy() * dt

        # Make object velocity relative to TCP
        object_velp_rel = object_velp - tcp_vel

        # Gripper state and velocity
        gripper_state = np.array([
            self.data.qpos[self.finger_qpos1_addr],
            self.data.qpos[self.finger_qpos2_addr],
        ])
        gripper_vel = np.array([
            self.data.qvel[self.finger_dof1_addr],
            self.data.qvel[self.finger_dof2_addr],
        ]) * dt

        obs = np.concatenate([
            tcp_pos,            # 3
            object_pos,         # 3
            object_rel_pos,     # 3
            gripper_state,      # 2
            object_rot,         # 3
            object_velp_rel,    # 3
            object_velr,        # 3
            tcp_vel,            # 3
            gripper_vel,        # 2
        ])  # total = 25

        return {
            "observation": obs.copy(),
            "achieved_goal": object_pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    # ==================================================================
    # Goal sampling
    # ==================================================================

    def _sample_goal(self):
        """Sample a goal position for the current episode.

        When randomize_goals=False:
            Returns a fixed goal 10cm in front of the object at table height.
            Gives the agent a single consistent task to master before adding
            variety. HER still relabels with achieved positions, providing
            goal diversity for free.

        When randomize_goals=True:
            Goal distribution (designed for progressive HER learning):
              30% near-object: within 10cm of object start — easy targets
              20% table-slide: broad XY range at table height
              50% lifted: workspace position above table

            MIN_GOAL_DISTANCE (8cm) prevents trivial initial successes.
        """
        obj_pos = self.data.xpos[self.package_body_id].copy()

        if not self.randomize_goals:
            # Fixed goal: 10cm in front of the object at table height.
            # Simple, repeatable task for initial skill acquisition.
            goal = obj_pos.copy()
            goal[0] += 0.10  # 10cm forward (toward robot base)
            goal = np.clip(goal, self.TCP_BOUNDS_LOW, self.TCP_BOUNDS_HIGH)
            return goal.copy()

        # --- Random goal sampling (lift-biased distribution) ---
        # Root cause fix: ~50% of former goals were at table height, causing the
        # base reward to PENALIZE lifting (object moves away from table-height goal).
        # New split: 10% near-object (HER bootstrap), 90% lifted workspace.
        for _ in range(100):
            r = self.np_random.random()

            if r < 0.10:
                # Near-object goals: provides easy HER curriculum for early learning.
                # Table height only — these teach the agent that moving the object
                # to a nearby XY position is rewarded.
                goal = obj_pos.copy()
                goal[0] += self.np_random.uniform(-0.10, 0.10)
                goal[1] += self.np_random.uniform(-0.10, 0.10)
                goal[2] = self.height_offset
            else:
                # Lifted workspace goals (90%): object must be raised to succeed.
                # Always at least 5cm above table so lifting is strictly required.
                goal = np.array([
                    0.50 + self.np_random.uniform(-0.15, 0.15),
                    0.00 + self.np_random.uniform(-0.25, 0.25),
                    self.height_offset,
                ])
                goal[2] += self.np_random.uniform(0.05, 0.20)

            goal = np.clip(goal, self.TCP_BOUNDS_LOW, self.TCP_BOUNDS_HIGH)

            if goal_distance(goal, obj_pos) >= self.MIN_GOAL_DISTANCE:
                return goal.copy()

        # Fallback: should virtually never reach here
        return goal.copy()

    # ==================================================================
    # Kinematics helpers
    # ==================================================================

    def _get_tcp_pos(self):
        """TCP position = hand body + rotated local offset to fingertip midpoint."""
        hand_pos = self.data.xpos[self.hand_body_id].copy()
        hand_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)
        return hand_pos + hand_mat @ self.TCP_LOCAL_OFFSET

    def _get_tcp_vel(self):
        """TCP translational velocity via full Jacobian * qvel."""
        jacp = np.zeros((3, self.model.nv))
        tcp_pos = self._get_tcp_pos()
        mujoco.mj_jac(
            self.model, self.data, jacp, None,
            np.ascontiguousarray(tcp_pos), self.hand_body_id,
        )
        return jacp @ self.data.qvel

    # ==================================================================
    # Visualization
    # ==================================================================

    def _update_target_viz(self):
        """Move the target visualization site to the current goal."""
        if self.target_site_id is not None:
            self.model.site_pos[self.target_site_id] = self.goal

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._update_target_viz()
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def _quat_to_euler(quat):
    """Convert MuJoCo quaternion [w, x, y, z] to Euler [roll, pitch, yaw]."""
    w, x, y, z = quat
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return np.array([roll, pitch, yaw])
