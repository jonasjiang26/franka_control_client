import os

import pyzlc
from scipy.spatial.transform import Rotation as R

from franka_control_client.core.exception import DeviceConnectionError
from franka_control_client.franka_robot.panda_arm import RemotePandaArm

if __name__ == "__main__":
    server_ip = os.getenv("PYZLC_IP", "141.3.53.25")
    group_name = os.getenv("PYZLC_GROUP", "robot_lab_robotiq_202")
    group_port = int(os.getenv("PYZLC_PORT", "7725"))
    robot_name = os.getenv("ROBOT_NAME", "FrankaPanda")

    pyzlc.init(
        "policy_inference",
        server_ip,
        group_name=group_name,
        group_port=group_port,
    )
    print(
        f"Connection config: ip={server_ip}, group_name={group_name}, group_port={group_port}, robot_name={robot_name}"
    )

    nodes = pyzlc.get_nodes_info(group_name)
    print("Visible nodes:", [node.name for node in nodes])
    for candidate in (
        robot_name,
        "FrankaPanda",
        "Panda201",
        "Panda202",
        "MujocoRobot",
    ):
        print(
            f"check_node_info('{candidate}') -> {pyzlc.check_node_info(candidate, group_name)}"
        )

    arm = RemotePandaArm(robot_name)
    try:
        arm.connect()
    except DeviceConnectionError as exc:
        print(exc)
        print(
            "Node lookup failed, but the state topic may still be reachable. Continuing in read-only mode."
        )
    
    while True:
        state = arm.current_state
        if state is not None:
            euler_xyz_deg = R.from_quat(state["EE_quat"]).as_euler(
                "xyz", degrees=True
            )
            print("Current end-effector position:", state["EE_pos"])
            print("Current end-effector rotation (quaternion):", state["EE_quat"])
            print(
                "Current end-effector rotation (Euler xyz, deg):",
                euler_xyz_deg,
            )
        else:
            print("No state received yet.")
        pyzlc.sleep(1)
