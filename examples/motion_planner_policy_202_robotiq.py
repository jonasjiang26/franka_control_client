from typing import List

import pyzlc

from franka_control_client.camera.camera import CameraDevice
from franka_control_client.control_pair.motion_planner_policy_panda_control_pair import (
    PolicyMotionPlannerControlPair
)

from franka_control_client.franka_robot.panda_arm import RemotePandaArm
from franka_control_client.franka_robot.panda_robotiq import PandaRobotiq
from franka_control_client.data_collection.irl_wrapper import (
    IRLDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    RobotiqGripperDataWrapper,
)
from franka_control_client.policy_inference.motion_planner_policy_inference import (
    MotionPlannerInference,
)
from franka_control_client.robotiq_gripper.robotiq_gripper import (
    RemoteRobotiqGripper,
)


if __name__ == "__main__":
    pyzlc.init(
        "policy_inference",
        "141.3.53.25",
        group_name="robot_lab_robotiq_202",
        group_port=7725,
    )

    #goal target to be grasped
    goal_prompt = "toy carrot."
    scene_prompt = "toy carrot. pan."

    follower = PandaRobotiq(
        "PandaRobotiq",
        RemotePandaArm("FrankaPanda"),
        RemoteRobotiqGripper("FrankaPanda"),
    )
    control_pair = PolicyMotionPlannerControlPair(
        follower.panda_arm, follower.robotiq_gripper, 500, 10, 0.05
    )

    data_collectors: List[IRLDataWrapper] = []
    data_collectors.append(PandaArmDataWrapper(follower.panda_arm))
    data_collectors.append(RobotiqGripperDataWrapper(follower.robotiq_gripper))

    inference_manager = MotionPlannerInference(
        data_collectors=data_collectors,
        control_pair=control_pair,
        task=goal_prompt,
        scene=scene_prompt,
        cfg=None,  # No additional config needed for this inference type
        )
    
    try:
        inference_manager.run()
    finally:
        pyzlc.shutdown()
