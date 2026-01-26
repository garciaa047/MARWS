"""View the warehouse scene in MuJoCo's native viewer."""
import mujoco
import mujoco.viewer
import time


def main():
    model = mujoco.MjModel.from_xml_path("simulation/assets/warehouse.xml")
    data = mujoco.MjData(model)

    # Set initial robot pose (arm slightly raised)
    data.qpos[0] = 0.0   # joint1
    data.qpos[1] = -0.5  # joint2
    data.qpos[2] = 1.0   # joint3
    data.qpos[3] = 0.5   # joint4
    data.qpos[4] = 0.0   # joint5
    data.qpos[5] = 0.0   # joint6

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close the window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
