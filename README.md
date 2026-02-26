# ARMS — Autonomous Robotic Manipulation System

A reinforcement learning project training a **Franka Emika Panda** robotic arm to autonomously pick and place objects in a simulated warehouse environment. Built from scratch using **SAC + HER** and **MuJoCo** physics simulation.

> **Current Status:** The arm has learned to consistently approach and grasp the target object. It has **not yet learned to lift or transport** the object to the goal position. Lift rate is currently 0%. Active development is ongoing — see [Current Progress](#current-progress).

<!-- Add a demo GIF here once available -->
<!-- ![ARMS Demo](assets/demo.gif) -->

---

## Project Highlights

- **Goal-conditioned reinforcement learning** using Hindsight Experience Replay (HER) with a dense reward function modelled on the OpenAI Fetch benchmark
- **Cartesian action space** controlled via real-time Jacobian IK — the agent controls 3D TCP displacement and gripper state rather than raw joint angles
- **Multi-stage reward engineering** with gated approach, grasp closure, TCP height incentive, and lift shaping, each designed to avoid creating local optima that trap the policy
- **VecNormalize** applied to all 25 observation dimensions — critical for convergence when features span four orders of magnitude
- **Trained entirely on CPU** at ~63 steps/second using PyTorch thread optimisation

---

## Architecture

### Algorithm

| Component | Choice | Rationale |
|---|---|---|
| Algorithm | SAC (Soft Actor-Critic) | Entropy-regularised; auto-tuned exploration prevents premature convergence |
| Goal strategy | HER (n=4, future) | Turns every failed episode into 4 synthetic successes via goal relabelling |
| Reward type | Dense: −‖achieved − desired‖ | Smooth critic gradients at every state; gradient always points toward goal |
| Action space | 4-dim Cartesian | Simpler than 8-dim joint space; kinematics handled analytically |
| Episode length | 100 steps | ~4 seconds of simulation time at 25 Hz control |

### Network

| Parameter | Value |
|---|---|
| Policy | MultiInputPolicy (Dict observation space) |
| Hidden layers | [256, 256] ReLU |
| Learning rate | 1 × 10⁻³ |
| Discount (γ) | 0.98 |
| Replay buffer | 1,000,000 transitions |
| Batch size | 256 |
| Entropy coefficient | Auto-tuned |
| Observation normalisation | VecNormalize (running mean / std) |

### Environment

| Property | Value |
|---|---|
| Simulator | MuJoCo (Python bindings) |
| Robot | Franka Emika Panda — 7-DOF arm + 2-finger parallel gripper |
| Object | 4 cm cube, 50 g, friction coefficient 1.0 |
| TCP workspace | X [0.25, 0.75] m, Y [−0.55, 0.55] m, Z [0.42, 0.80] m |
| Goal distribution | 10% near-object at table height, 90% lifted 5–20 cm above table |
| Kinematics | Damped Jacobian pseudoinverse IK (λ = 1 × 10⁻⁴) |
| Control rate | 50 Hz (10 substeps × 0.002 s timestep) |

### Reward Structure

The reward is split between a HER-compatible base signal and step-only shaping terms. The shaping terms live exclusively in `step()` and are never applied to HER's relabelled virtual transitions, which use only `compute_reward()`.

| Term | Active when | Purpose |
|---|---|---|
| Base: −‖object − goal‖ | Always (via `compute_reward`) | Smooth gradient toward goal for critic |
| XY approach | Always | Pull TCP horizontally over object |
| Z descent | XY < 5 cm **and** grip > 5 cm | Force top-down approach geometry; gate prevents hover optimum |
| Grasp closure | grip < 5 cm **and** fingers closing | Incentivise maintaining grip when in position |
| TCP height incentive | grip < 5 cm **and** fingers closing | Direct upward gradient for arm's z action before box moves |
| Lift shaping | grip < 5 cm **and** box rising **and** goal above box | Reinforce actual upward object movement |

---

## Current Progress

> **The arm has not yet achieved lift.** This is explicitly stated to set accurate expectations.

The arm has mastered the approach phase:

- Consistently moves its TCP to within **2.5–4 cm** of the object from ~150k training steps onward
- Correctly approaches from above — top-down geometry enforced by staged reward shaping
- Sometimes achieves a physical grasp on the object

What it has not yet learned:

- **Lifting** — the object has never been raised off the table in any training run to date
- **Transporting** — the goal position has never been reached
- **Consistent hold** — gripper closure behaviour is currently stochastic

The core difficulty is providing the arm with gradient toward vertical movement. Once the arm is close to the object, the reward landscape is flat in the z direction unless the box is already moving — a chicken-and-egg problem that requires careful reward design to resolve. See [Training Challenges](#training-challenges).

---

## Getting Started

### Prerequisites

- Python 3.10+
- MuJoCo (`pip install mujoco`)
- Stable Baselines3 with extras

### Installation

```bash
git clone https://github.com/garciaa047/MARWS.git
cd MARWS
pip install stable-baselines3[extra] mujoco tensorboard tqdm rich
```

### Training

```bash
# Standard training run (2M steps, ~9 hours on CPU)
python -m training.train_single_agent --timesteps 2000000

# Resume from a saved checkpoint
python -m training.train_single_agent --resume models/sac_her/latest_model.zip

# Enable object position randomisation (off by default)
python -m training.train_single_agent --randomize-object

# Monitor training in TensorBoard
tensorboard --logdir models/sac_her/logs
```

### Evaluation

```bash
# Evaluate the best saved model (headless)
python -m scripts.evaluate --model models/sac_her/best_model.zip

# Evaluate with live rendering
python -m scripts.evaluate --model models/sac_her/best_model.zip --render
```

Training saves checkpoints every 50,000 steps to `models/sac_her/`. The best model by evaluation reward is saved as `best_model.zip`. A graceful shutdown (Ctrl+C) saves the current model and normalisation stats before exiting.

---

## Training Metrics

All metrics are logged to TensorBoard and printed to stdout every 5,000 steps.

| Metric | Description |
|---|---|
| `marws/success_rate` | Fraction of episodes where the object reached the goal position |
| `marws/lift_rate` | Fraction of episodes where the object rose more than 3 cm above the table |
| `marws/mean_grip_obj_dist` | Mean 3D distance from TCP to object — primary measure of approach progress |
| `marws/mean_distance` | Mean final object-to-goal distance at episode end |
| `marws/mean_episode_reward` | Mean total episode reward over the last 100 episodes |
| `marws/mean_goal_height` | Mean sampled goal height — confirms goal distribution is correct |

---

## Training Challenges

Designing the reward function for this task was the central engineering challenge of the project. This section documents the problems encountered and the reasoning behind each fix, as a record of what was learned.

### Why dense reward alone is not enough

The base reward `−‖object − goal‖` provides no gradient for the arm's actions when the object is not moving. The arm can only improve this reward by physically lifting the box — but it receives no signal telling it how to do that. Any reward term anchored to the object's position has the same fundamental problem: the gradient with respect to the arm's action is zero when the object is stationary.

```
∂r/∂action_z = (∂r/∂object_z) × (∂object_z/∂action_z) = non-zero × 0 = 0
```

This means approaches like "penalise the z-distance between object and goal" do not teach the arm to lift — they add a constant penalty that does not change regardless of what the arm does in z.

### The hover local optimum

The first reward structure used staged shaping: pull the arm horizontally over the object (XY alignment), then descend to object height (Z descent). Z descent created a stable hover optimum — the arm reached object height and stayed there, because any upward movement was immediately penalised. Episode reward plateaued at approximately −19 to −21 from ~150k steps onward and never improved regardless of training duration.

**Fix:** Z descent is gated off once the arm is within 5 cm of the object (inside the grasp zone). The arm can now move upward without penalty once it is in position.

### The discovery problem

Removing Z descent inside the grasp zone eliminated the downward anchor but did not add any upward signal. The reward landscape inside the grasp zone remained completely flat in the z direction. Lift shaping (rewarding positive delta in object z) cannot fire until the box is already moving — requiring the arm to have already discovered and executed a successful grasp and lift before receiving any reinforcement for doing so.

A grasp closure bonus and increased lift shaping weight were added to strengthen the signal once a grasp occurs, but at 500k+ steps the lift rate remained 0%. The arm had no reason to attempt vertical movement in the first place.

**Fix (in progress):** A TCP height incentive rewards the arm's TCP position approaching goal height when inside the grasp zone with a closed gripper. This provides direct upward gradient based on the arm's own position — not the object's — so the gradient is non-zero even when the box has not yet moved. A self-correction mechanism prevents open-gripper flight: if the arm rises without the box, it exits the grasp zone, Z descent reactivates, and the arm is pulled back.

### Key insight: gradient lives in the arm's position space, not the object's

The most important lesson from reward debugging was distinguishing between two superficially similar reward terms:

- `−W × |object_z − goal_z|` — gradient zero when box is stationary; arm cannot affect this
- `−W × |tcp_z − goal_z|` — gradient non-zero always; arm directly controls its own z position

Both look like "penalise z-distance to goal" but only one provides gradient for the arm's actions.

---

## What I Learned

**HER fundamentally changes how reward must be structured.** Because HER relabels 80% of transitions using only `compute_reward()`, any shaping reward that should not apply to relabelled transitions must live exclusively in `step()`. Mixing the two corrupts the critic's value estimates for the relabelled experience.

**SAC is powerful but fragile in shaped reward environments.** SAC with HER can learn consistent approach behaviour from scratch in under 200k steps — but a single reward design error can cause the Q-function to converge to an incorrect value landscape that is very difficult to escape without restarting training entirely.

**VecNormalize is not optional for this task.** The 25-dimensional observation spans four orders of magnitude: joint velocities are on the order of 0.001 while Cartesian positions are on the order of 0.5. Without running normalisation the critic diverges within the first 50k steps.

**Reward gradient and reward magnitude are separate problems.** Increasing a reward coefficient does not help if the gradient of that reward with respect to the arm's action is already zero. Several attempted fixes increased coefficient values without changing which variables the reward depended on — and therefore changed nothing about the arm's behaviour.

**Local optima are often more stable than they appear.** The hover policy was not merely a suboptimal solution — it was the Q-function's correct answer to a poorly specified reward function. Understanding why required tracing the exact gradient landscape at the hover state, not just observing the trained behaviour in the evaluator.

**Overestimating initial scope is a real risk in robotics projects.** This project was originally designed as a multi-agent warehouse coordination system. Reducing it to single-agent pick-and-place revealed how much complexity exists even in the most fundamental manipulation task. Getting a robot arm to reliably pick up a box is genuinely hard.

---

## Known Limitations

- **No lift yet** — the arm approaches and sometimes grasps the object but has not learned to lift it. This is the current development priority.
- **CPU-only training** — no GPU available in the current environment. Throughput is ~63 steps/second; a 2M-step run takes approximately 9 hours.
- **No object position randomisation** — the object starts at a fixed position each episode. Randomisation is implemented but disabled by default until lift is achieved.
- **No environment parallelisation** — SB3's `HerReplayBuffer` only supports `n_envs=1`, preventing parallel environment collection to increase throughput.
- **Single agent only** — the project was originally scoped for multi-agent warehouse coordination but was reduced in scope.

---

## Future Plans

- [ ] Achieve consistent lift and transport to goal position *(current priority)*
- [ ] Enable object position randomisation once lift is stable
- [ ] Extend goal sampling across a wider workspace volume
- [ ] Evaluate sim-to-real transfer potential
- [ ] Revisit multi-agent coordination once single-agent task is solved

---

## Project Structure

```
ARMS/
├── simulation/
│   ├── env.py                      # Goal-conditioned MuJoCo environment (GoalEnv)
│   └── franka_emika_panda/
│       ├── warehouse_scene.xml     # Scene: table, object, goal marker
│       └── panda.xml               # Franka Emika Panda MJCF model
├── training/
│   ├── config.py                   # SAC + HER hyperparameters
│   └── train_single_agent.py       # Training script with callbacks and logging
├── scripts/
│   ├── evaluate.py                 # Model evaluation across episodes
│   └── view_scene.py               # MuJoCo scene viewer
├── models/
│   └── sac_her/                    # Saved models, checkpoints, TensorBoard logs
│       ├── best_model.zip          # Best model by evaluation reward
│       ├── latest_model.zip        # Most recent checkpoint
│       └── vecnormalize.pkl        # Observation normalisation statistics
├── tests/
│   └── test_env.py                 # Environment unit tests
├── REWARD_DESIGN.md                # Reward function design documentation
└── PROJECT_PLAN.md                 # Project roadmap
```

---

## Built With

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — SAC + HER implementation
- [MuJoCo](https://mujoco.org/) — Physics simulation
- [Gymnasium](https://gymnasium.farama.org/) — RL environment interface
- [TensorBoard](https://www.tensorflow.org/tensorboard) — Training visualisation
- [PyTorch](https://pytorch.org/) — Neural network backend

---

*This project was originally named MARWS (Multi-Agent Robotic Warehouse System). The scope was reduced to single-agent pick-and-place after the complexity of the base manipulation task became clear. The repository name will be updated in a future release.*
