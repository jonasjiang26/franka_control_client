from collections import deque
import gc
from threading import Lock
from typing import Any, Deque, List, Optional

import pyzlc
import torch

from franka_control_client.camera.camera import CameraDevice
from franka_control_client.control_pair.cartesian_policy_panda_control_pair import (
    PolicyPandaRobotiqDeltaCartesianControlPair,
)
from franka_control_client.data_collection.irl_wrapper import (
    IRLDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    RobotiqGripperDataWrapper,
)
from franka_control_client.franka_robot.panda_arm import RemotePandaArm
from franka_control_client.franka_robot.panda_robotiq import PandaRobotiq
from franka_control_client.policy_inference.lerobot_policy_inference import (
    LeRobotPolicyInferenceConfig,
)
from franka_control_client.policy_inference.mq3_traj_visual_lerobot_inference import (
    MQ3TrajVisualLeRobotInference,
)
from franka_control_client.robotiq_gripper.robotiq_gripper import (
    RemoteRobotiqGripper,
)

EVAL_CONTROL_TOPIC = "policy2eval/control"
EVAL_STATUS_TOPIC = "policy2eval/status"


class Policy2EvalInference:
    def __init__(
        self,
        data_collectors: List[IRLDataWrapper],
        control_pair: PolicyPandaRobotiqDeltaCartesianControlPair,
        cfg: LeRobotPolicyInferenceConfig,
        control_topic: str = EVAL_CONTROL_TOPIC,
        status_topic: str = EVAL_STATUS_TOPIC,
    ) -> None:
        self.data_collectors = data_collectors
        self.control_pair = control_pair
        self.cfg = cfg
        self._active_inference: Optional[MQ3TrajVisualLeRobotInference] = None
        self._exiting = False
        self._command_queue: Deque[str] = deque()
        self._command_lock = Lock()
        self._status_pub = pyzlc.Publisher(status_topic)
        pyzlc.register_subscriber_handler(control_topic, self._on_control_signal)

    def _on_control_signal(self, payload: Any) -> None:
        if isinstance(payload, dict):
            command = str(payload.get("command", "")).lower()
        else:
            command = str(payload).lower()
        if command not in {"start", "stop", "quit"}:
            pyzlc.warning(f"Ignoring unknown eval control command: {payload!r}")
            return
        with self._command_lock:
            self._command_queue.append(command)

    def _pop_commands(self) -> list[str]:
        with self._command_lock:
            commands = list(self._command_queue)
            self._command_queue.clear()
        return commands

    def _publish_status(self, status: str, error: Optional[str] = None) -> None:
        payload = {
            "status": status,
            "active": self._active_inference is not None,
            "task": self.cfg.task,
        }
        if error is not None:
            payload["error"] = error
        self._status_pub.publish(
            payload
        )

    def _handle_command(self, command: str) -> None:
        if command == "start":
            self._start_eval_policy()
            return

        if command == "stop":
            self._stop_eval_policy()
            return

        if command == "quit":
            self._stop_eval_policy()
            self._exiting = True
            self._publish_status("exiting")

    def _start_eval_policy(self) -> None:
        if self._active_inference is not None:
            self._publish_status("already_started")
            return

        self._publish_status("loading")
        try:
            inference = MQ3TrajVisualLeRobotInference(
                data_collectors=self.data_collectors,
                control_pair=self.control_pair,
                cfg=self.cfg,
            )
            self._active_inference = inference
            inference._start_infering()
            self._publish_status("started")
        except Exception as exc:
            self._release_eval_policy()
            pyzlc.error(f"Failed to start eval policy: {exc}")
            self._publish_status("start_failed", error=str(exc))
            raise

    def _stop_eval_policy(self) -> None:
        if self._active_inference is None:
            self._publish_status("ready")
            return

        try:
            self._active_inference._discard_infering()
        finally:
            self._release_eval_policy()
            self._publish_status("stopped")

    def _release_eval_policy(self) -> None:
        inference = self._active_inference
        self._active_inference = None
        if inference is not None:
            try:
                inference._close()
            except Exception as exc:
                pyzlc.warning(f"Error while closing eval policy: {exc}")
            del inference

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError as exc:
                pyzlc.warning(f"CUDA IPC cache collection skipped: {exc}")
        self.control_pair.reset_action()

    def run(self) -> None:
        self._publish_status("ready")
        try:
            while not self._exiting:
                for command in self._pop_commands():
                    self._handle_command(command)

                if self._active_inference is not None:
                    self._active_inference._infer_step()

                pyzlc.sleep(0.001)
        finally:
            self._stop_eval_policy()


if __name__ == "__main__":
    pyzlc.init(
        "policy2eval_inference",
        "141.3.53.25",
        group_name="robot_lab_robotiq_202",
        group_port=7725,
    )

    eval_checkpoint_path = (
        "/home/jjiang/jing/model/beso/carrot_in_pan/checkpoints/003000/pretrained_model"
    )
    eval_task = "put the toy carrot in the pan."
    eval_dataset_path = "/home/jjiang/jing/dataset/lerobot/carrot_in_pan_trimmed"

    follower = PandaRobotiq(
        "PandaRobotiq",
        RemotePandaArm("FrankaPanda"),
        RemoteRobotiqGripper("FrankaPanda"),
    )
    control_pair = PolicyPandaRobotiqDeltaCartesianControlPair(
        follower.panda_arm, follower.robotiq_gripper, 500, 10, 0.05
    )

    static_cam = ImageDataWrapper(CameraDevice("static_cam", preview=False), hw_name="image")
    wrist_cam = ImageDataWrapper(CameraDevice("wrist_cam", preview=False), hw_name="image2")

    data_collectors: List[IRLDataWrapper] = [
        static_cam,
        wrist_cam,
        PandaArmDataWrapper(follower.panda_arm),
        RobotiqGripperDataWrapper(follower.robotiq_gripper),
    ]

    eval_inference_cfg = LeRobotPolicyInferenceConfig(
        checkpoint_path=eval_checkpoint_path,
        task=eval_task,
        fps=20,
        device="cuda",
        dataset_path=eval_dataset_path,
    )

    inference_manager = Policy2EvalInference(
        data_collectors=data_collectors,
        control_pair=control_pair,
        cfg=eval_inference_cfg,
    )

    try:
        inference_manager.run()
    finally:
        pyzlc.shutdown()
