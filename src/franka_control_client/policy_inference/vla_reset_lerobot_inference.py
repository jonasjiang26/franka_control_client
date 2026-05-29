from typing import Any, Dict, List, Optional

import numpy as np

from ..control_pair.cartesian_policy_panda_control_pair import (
    PolicyPandaRobotiqDeltaCartesianControlPair,
)
from ..data_collection.irl_wrapper import IRLDataWrapper
from .lerobot_policy_inference import LeRobotPolicyInferenceConfig
from .mq3_traj_visual_lerobot_inference import MQ3TrajVisualLeRobotInference
from .policy_inference_manager import PolicyInferenceEvent, PolicyInferenceState

GRIPPER_RELEASE_THRESHOLD = 0.5


class VLAResetLeRobotInference(MQ3TrajVisualLeRobotInference):
    def __init__(
        self,
        data_collectors: List[IRLDataWrapper],
        control_pair: PolicyPandaRobotiqDeltaCartesianControlPair,
        eval_cfg: LeRobotPolicyInferenceConfig,
        reset_cfg: LeRobotPolicyInferenceConfig,
    ) -> None:
        super().__init__(data_collectors, control_pair, eval_cfg)
        self.eval_cfg = eval_cfg
        self.reset_cfg = reset_cfg
        self._policy_bundles: Dict[str, Dict[str, Any]] = {
            "eval": self._capture_policy_bundle(eval_cfg)
        }
        self._policy_bundles["reset"] = self._load_policy_bundle(reset_cfg)
        self._active_policy_name = "eval"
        self._next_start_policy_name: Optional[str] = None
        self._restore_policy_bundle(self._policy_bundles["eval"])
        self._reset_release_watch()
        self._awaiting_reset_confirmation = False
        self._awaiting_eval_confirmation = False

    def _capture_policy_bundle(
        self,
        cfg: LeRobotPolicyInferenceConfig,
    ) -> Dict[str, Any]:
        return {
            "cfg": cfg,
            "task": cfg.task,
            "fps": cfg.fps,
            "train_cfg": self.train_cfg,
            "policy": self.policy,
            "preprocessor": self.preprocessor,
            "postprocessor": self.postprocessor,
            "expected_image_shapes": self._expected_image_shapes,
            "expected_state_dim": self._expected_state_dim,
        }

    def _restore_policy_bundle(self, bundle: Dict[str, Any]) -> None:
        self.cfg = bundle["cfg"]
        self.task = bundle["task"]
        self.fps = bundle["fps"]
        self.train_cfg = bundle["train_cfg"]
        self.policy = bundle["policy"]
        self.preprocessor = bundle["preprocessor"]
        self.postprocessor = bundle["postprocessor"]
        self._expected_image_shapes = bundle["expected_image_shapes"]
        self._expected_state_dim = bundle["expected_state_dim"]

    def _load_policy_bundle(
        self,
        cfg: LeRobotPolicyInferenceConfig,
    ) -> Dict[str, Any]:
        current_bundle = self._capture_policy_bundle(self.cfg)
        self.cfg = cfg
        self.task = cfg.task
        self.fps = cfg.fps
        self.train_cfg = self._load_train_cfg()
        self.policy, self.preprocessor, self.postprocessor = (
            self._load_policy_stack()
        )
        self._expected_image_shapes = self._get_expected_image_shapes()
        self._expected_state_dim = self._get_expected_state_dim()
        bundle = self._capture_policy_bundle(cfg)
        self._restore_policy_bundle(current_bundle)
        return bundle

    def _activate_policy(self, policy_name: str) -> None:
        self._active_policy_name = policy_name
        self._restore_policy_bundle(self._policy_bundles[policy_name])
        self._reset_release_watch()

    def _reset_release_watch(self) -> None:
        self._saw_closed_gripper = False
        self._first_release_detected = False

    def _handle_policy_action(self, action_vec: np.ndarray) -> None:
        self._update_release_watch()
        super()._handle_policy_action(action_vec)

    def _get_gripper_sensor_value(self) -> Optional[float]:
        if self.gripper_wrapper is None:
            return None

        grip_state = self.gripper_wrapper.capture_step()
        if not isinstance(grip_state, dict):
            return None

        if "position" in grip_state:
            return float(grip_state["position"])
        if "width" in grip_state:
            return float(grip_state["width"])
        if "gripper" in grip_state:
            gripper_arr = np.asarray(
                grip_state["gripper"], dtype=np.float32
            ).reshape(-1)
            if gripper_arr.size > 0:
                return float(gripper_arr[0])
        return None

    def _update_release_watch(self) -> None:
        if self._first_release_detected:
            return

        gripper_value = self._get_gripper_sensor_value()
        if gripper_value is None:
            return

        gripper_closed = gripper_value >= GRIPPER_RELEASE_THRESHOLD
        if gripper_closed:
            self._saw_closed_gripper = True
            return
        if not self._saw_closed_gripper:
            return

        self._first_release_detected = True
        if self._active_policy_name == "eval":
            self._awaiting_reset_confirmation = True
            self._ui_console.update_hint(
                "Eval gripper released. Press 'c' to begin scene reset, 'd' to discard, or 'q' to quit"
            )
        elif self._active_policy_name == "reset":
            self._awaiting_eval_confirmation = True
            self._ui_console.update_hint(
                "Reset gripper released. Press 'e' to begin a new eval episode, 'd' to discard, or 'q' to quit"
            )

    def _handle_custom_keypress(self, key: str) -> bool:
        if key == "d" and self._state_machine.state == PolicyInferenceState.INFERING:
            self._clear_policy_switch_state()
            self._state_machine.trigger(PolicyInferenceEvent.DISCARD)
            return True
        if (
            key == "c"
            and self._awaiting_reset_confirmation
            and self._state_machine.state == PolicyInferenceState.INFERING
        ):
            self._start_reset_policy()
            return True
        if (
            key == "e"
            and self._awaiting_eval_confirmation
            and self._state_machine.state == PolicyInferenceState.INFERING
        ):
            self._start_eval_policy()
            return True
        return False

    def _clear_policy_switch_state(self) -> None:
        self._next_start_policy_name = None
        self._awaiting_reset_confirmation = False
        self._awaiting_eval_confirmation = False
        self._reset_release_watch()

    def _discard_infering(self) -> None:
        self._clear_policy_switch_state()
        super()._discard_infering()

    def _start_reset_policy(self) -> None:
        self._awaiting_reset_confirmation = False
        self._stop_infering()
        self._reset_arm()
        self._next_start_policy_name = "reset"
        self._start_infering()

    def _start_eval_policy(self) -> None:
        self._awaiting_eval_confirmation = False
        self._stop_infering()
        self._reset_arm()
        self._next_start_policy_name = "eval"
        self._start_infering()

    def _start_infering(self) -> None:
        self._activate_policy(self._next_start_policy_name or "eval")
        self._next_start_policy_name = None
        self._awaiting_reset_confirmation = False
        self._awaiting_eval_confirmation = False
        super()._start_infering()
