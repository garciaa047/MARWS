"""View the warehouse scene in MuJoCo's native viewer."""
import mujoco
import mujoco.viewer
import numpy as np
import time

TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.103])


def main():
    model = mujoco.MjModel.from_xml_path("simulation/franka_emika_panda/warehouse_scene.xml")
    data = mujoco.MjData(model)

    hand_id = model.body("hand").id
    marker_mocap_id = model.body_mocapid[model.body("tcp_marker").id]

    # Home configuration matching env._setup_initial_state()
    data.qpos[0] =  0.0      # joint1
    data.qpos[1] =  0.0      # joint2
    data.qpos[2] =  0.0      # joint3
    data.qpos[3] = -1.57079  # joint4  (-pi/2)
    data.qpos[4] =  0.0      # joint5
    data.qpos[5] =  1.57079  # joint6  (+pi/2)
    data.qpos[6] = -0.7853   # joint7  (-pi/4)
    data.ctrl[7] = 255       # gripper open

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close the window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            tcp = data.xpos[hand_id] + data.xmat[hand_id].reshape(3, 3) @ TCP_LOCAL_OFFSET
            data.mocap_pos[marker_mocap_id] = tcp
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
