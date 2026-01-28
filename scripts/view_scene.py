"""View the warehouse scene in MuJoCo's native viewer."""
import mujoco
import mujoco.viewer
import time


def main():
    model = mujoco.MjModel.from_xml_path("simulation/franka_emika_panda/warehouse_scene.xml")
    data = mujoco.MjData(model)

    # Set initial robot pose (arm slightly raised)
    data.qpos[0] = 0.0   # joint1
    data.qpos[1] = 0.0  # joint2
    data.qpos[2] = 0.0   # joint3
    data.qpos[3] = -1.5   # joint4
    data.qpos[4] = 0.0   # joint5
    data.qpos[5] = 1.5   # joint6
    data.qpos[6] = -0.7  # joint7
    # 0, 0, 0, -1.57079, 0, 1.57079, -0.7853

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close the window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
