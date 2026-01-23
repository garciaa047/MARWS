"""Diagnostic script to understand why grasping isn't working."""
import numpy as np
import mujoco
from simulation.env import MarwsEnv


def run_diagnostic(num_steps=500):
    """Run environment with random actions and print diagnostic info."""
    env = MarwsEnv(render_mode="human", max_steps=num_steps)
    obs, _ = env.reset()

    print("=" * 60)
    print("GRASPING DIAGNOSTIC")
    print("=" * 60)

    # Print initial state
    gripper_pos = env._get_gripper_position()
    package_pos = env._get_package_position()
    print(f"\nInitial gripper position: {gripper_pos}")
    print(f"Initial package position: {package_pos}")
    print(f"Initial distance: {np.linalg.norm(gripper_pos - package_pos):.4f}m")

    closest_dist = float('inf')
    grasp_attempts = 0
    contacts_detected = 0

    for step in range(num_steps):
        # Create action that moves toward package then tries to grasp
        gripper_pos = env._get_gripper_position()
        package_pos = env._get_package_position()
        dist = np.linalg.norm(gripper_pos - package_pos)

        # Random actions for joints, but controlled gripper
        action = np.random.uniform(-0.3, 0.3, 7)

        # If close to package, close gripper
        if dist < 0.1:
            action[6] = -1.0  # Close gripper
            grasp_attempts += 1
        else:
            action[6] = 1.0  # Open gripper

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        # Track closest approach
        if dist < closest_dist:
            closest_dist = dist

        # Check contacts
        contacts = env._get_gripper_contacts()
        if len(contacts) > 0:
            contacts_detected += 1

        # Print periodic diagnostics
        if step % 100 == 0 or len(contacts) > 0 or env.gripper_holding:
            gripper_joint = env.data.qpos[6]
            print(f"\nStep {step}:")
            print(f"  Distance to package: {dist:.4f}m")
            print(f"  Gripper joint pos: {gripper_joint:.4f} (closed < 0.02)")
            print(f"  Gripper action: {action[6]:.2f}")
            print(f"  Contacts: {len(contacts)}")
            print(f"  Holding: {env.gripper_holding}")
            print(f"  Reward: {reward:.4f}")

        if env.gripper_holding:
            print("\n*** SUCCESSFUL GRASP! ***")

        if terminated or truncated:
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Closest approach: {closest_dist:.4f}m")
    print(f"Grasp attempts (close + gripper closing): {grasp_attempts}")
    print(f"Steps with contacts: {contacts_detected}")
    print(f"Final holding state: {env.gripper_holding}")

    env.close()


def test_manual_grasp():
    """Test if grasping is physically possible with perfect positioning."""
    print("\n" + "=" * 60)
    print("MANUAL GRASP TEST")
    print("=" * 60)

    env = MarwsEnv(render_mode="human", max_steps=1000)
    obs, _ = env.reset()

    package_pos = env._get_package_position()
    print(f"Package position: {package_pos}")
    print(f"Package is at height z={package_pos[2]:.3f}m")

    # Try to manually position gripper above package
    print("\nAttempting to position gripper above package...")
    print("Sending actions to move toward package position...")

    for step in range(500):
        gripper_pos = env._get_gripper_position()
        package_pos = env._get_package_position()

        # Simple proportional control toward package
        direction = package_pos - gripper_pos
        direction[2] += 0.05  # Aim slightly above package

        # Normalize and scale
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            direction = direction / dist * 0.5

        # Create action (this is simplified - real IK would be needed)
        action = np.zeros(7)
        action[6] = 1.0 if dist > 0.08 else -1.0  # Open when far, close when near

        # Random small joint movements (not real control, just for testing)
        action[:6] = np.random.uniform(-0.2, 0.2, 6)

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if step % 50 == 0:
            print(f"Step {step}: dist={dist:.3f}, gripper_joint={env.data.qpos[6]:.3f}, contacts={len(env._get_gripper_contacts())}")

        if env.gripper_holding:
            print(f"\n*** GRASP ACHIEVED at step {step}! ***")
            break

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    print("Running grasping diagnostics...\n")
    run_diagnostic(300)

    input("\nPress Enter to run manual grasp test...")
    test_manual_grasp()
