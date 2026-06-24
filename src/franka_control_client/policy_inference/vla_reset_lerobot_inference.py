import pprint
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pyzlc

from ..control_pair.cartesian_policy_panda_control_pair import (
    PolicyPandaRobotiqDeltaCartesianControlPair,
)
from ..data_collection.irl_wrapper import IRLDataWrapper
from .lerobot_policy_inference import LeRobotPolicyInferenceConfig
from .mq3_traj_visual_lerobot_inference import MQ3TrajVisualLeRobotInference
from .policy_inference_manager import PolicyInferenceEvent, PolicyInferenceState

GRIPPER_RELEASE_THRESHOLD = 0.5
PYZLC_GROUP_NAME = "robot_lab_robotiq_202"
SPATIAL_RELATION_SERVICE_NAME = "scene_graph"
SPATIAL_RELATION_REQUEST_INTERVAL = 6.0
SPATIAL_RELATION_REQUEST_TIMEOUT = 60.0


class VLAResetLeRobotInference(MQ3TrajVisualLeRobotInference):
    def __init__(
        self,
        data_collectors: List[IRLDataWrapper],
        control_pair: PolicyPandaRobotiqDeltaCartesianControlPair,
        eval_cfg: LeRobotPolicyInferenceConfig,
        reset_cfg: LeRobotPolicyInferenceConfig,
        items: str
    ) -> None:
        eval_cfg.lazy_load_policy = True
        reset_cfg.lazy_load_policy = True
        super().__init__(data_collectors, control_pair, eval_cfg)
        self.eval_cfg = eval_cfg
        self.reset_cfg = reset_cfg
        self._policy_bundles: Dict[str, Dict[str, Any]] = {
            "eval": self._capture_policy_bundle(eval_cfg)
        }
        self._policy_bundles["reset"] = self._load_policy_bundle(reset_cfg)
        self._active_policy_name = "eval"
        self.item_prompt = items
        self._next_start_policy_name: Optional[str] = None
        self._restore_policy_bundle(self._policy_bundles["eval"])
        self._reset_release_watch()
        self._awaiting_reset_confirmation = False
        self._awaiting_eval_confirmation = False
        self._pyzlc_group_name = PYZLC_GROUP_NAME
        self._spatial_relation_service_name = SPATIAL_RELATION_SERVICE_NAME
        self._spatial_relation_request_interval = (
            SPATIAL_RELATION_REQUEST_INTERVAL
        )
        self._spatial_relation_request_timeout = (
            SPATIAL_RELATION_REQUEST_TIMEOUT
        )
        self._next_spatial_relation_request_ts = 0.0
        self._spatial_relation_request_lock = threading.Lock()
        self._spatial_relation_request_in_flight = False
        self._pending_spatial_relation_request_reason: Optional[str] = None
        self._pending_spatial_relation_force_new = False
        self._spatial_relation_request_id: Optional[str] = None
        self._spatial_relation_request_prompt: Optional[str] = None

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
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._expected_image_shapes = self._get_expected_image_shapes()
        self._expected_state_dim = self._get_expected_state_dim()
        bundle = self._capture_policy_bundle(cfg)
        self._restore_policy_bundle(current_bundle)
        return bundle

    def _activate_policy(self, policy_name: str) -> None:
        self._active_policy_name = policy_name
        self._restore_policy_bundle(self._policy_bundles[policy_name])
        self._reset_release_watch()
        self._next_spatial_relation_request_ts = 0.0

    def _reset_release_watch(self) -> None:
        self._saw_closed_gripper = False
        self._first_release_detected = False

    def _handle_policy_action(self, action_vec: np.ndarray) -> None:
        self._maybe_request_spatial_relation()
        if self._awaiting_reset_confirmation or self._awaiting_eval_confirmation:
            return
        if self._update_release_watch():
            return
        super()._handle_policy_action(action_vec)

    def _maybe_request_spatial_relation(self) -> None:
        now = time.monotonic()
        if now < self._next_spatial_relation_request_ts:
            return

        self._next_spatial_relation_request_ts = (
            now + self._spatial_relation_request_interval
        )
        self._request_spatial_relation_async(
            f"{self._active_policy_name} periodic",
            force_new=False,
        )

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

    def _update_release_watch(self) -> bool:
        if self._first_release_detected:
            return False

        gripper_value = self._get_gripper_sensor_value()
        if gripper_value is None:
            return False

        gripper_closed = gripper_value >= GRIPPER_RELEASE_THRESHOLD
        if gripper_closed:
            self._saw_closed_gripper = True
            return False
        if not self._saw_closed_gripper:
            return False

        self._first_release_detected = True
        if self._active_policy_name == "eval":
            self._handle_eval_release()
            return True
        elif self._active_policy_name == "reset":
            self._handle_reset_release()
            return True
        return False

    def _handle_eval_release(self) -> None:
        self._request_spatial_relation_async(
            "eval gripper release",
            force_new=True,
        )
        self._awaiting_reset_confirmation = False
        self._awaiting_eval_confirmation = False
        message = "Eval released. Scene graph requested; eval policy continues."
        print(message, flush=True)
        pyzlc.info(message)
        self._ui_console.log(message)

    def _handle_reset_release(self) -> None:
        self._request_spatial_relation_async(
            "reset gripper release",
            force_new=True,
        )
        self._awaiting_reset_confirmation = False
        self._awaiting_eval_confirmation = False
        message = "Reset released. Scene graph requested; reset policy continues."
        print(message, flush=True)
        pyzlc.info(message)
        self._ui_console.log(message)

    def _request_spatial_relation_async(
        self,
        reason: str,
        force_new: bool,
    ) -> None:
        with self._spatial_relation_request_lock:
            if self._spatial_relation_request_in_flight:
                self._pending_spatial_relation_request_reason = reason
                self._pending_spatial_relation_force_new = (
                    self._pending_spatial_relation_force_new or force_new
                )
                return

            if force_new or self._spatial_relation_request_id is None:
                safe_reason = "_".join(reason.split())
                self._spatial_relation_request_id = (
                    f"{self._active_policy_name}-{safe_reason}-"
                    f"{uuid.uuid4().hex}"
                )
                self._spatial_relation_request_prompt = self.task

            request_id = self._spatial_relation_request_id
            prompt = self.item_prompt
            self._spatial_relation_request_in_flight = True

        thread = threading.Thread(
            target=self._spatial_relation_request_worker,
            args=(reason, request_id, prompt),
            daemon=True,
        )
        thread.start()

    def _spatial_relation_request_worker(
        self,
        trigger_reason: str,
        request_id: Optional[str],
        prompt: str,
    ) -> None:
        request_complete = False
        try:
            response = self._request_spatial_relation(
                request_id=request_id,
                prompt=prompt,
            )
            self._print_spatial_relation_response(trigger_reason, response)
            request_complete = self._spatial_relation_response_complete(
                response
            )
        finally:
            pending_reason = None
            pending_force_new = False
            with self._spatial_relation_request_lock:
                self._spatial_relation_request_in_flight = False
                if (
                    request_complete
                    and request_id == self._spatial_relation_request_id
                ):
                    self._spatial_relation_request_id = None
                    self._spatial_relation_request_prompt = None
                pending_reason = self._pending_spatial_relation_request_reason
                pending_force_new = self._pending_spatial_relation_force_new
                self._pending_spatial_relation_request_reason = None
                self._pending_spatial_relation_force_new = False

            if pending_reason is not None:
                self._request_spatial_relation_async(
                    pending_reason,
                    force_new=pending_force_new,
                )

    def _request_spatial_relation(
        self,
        request_id: Optional[str],
        prompt: str,
    ) -> Any:
        if request_id is None:
            return {
                "success": False,
                "message": "missing scene graph request_id",
            }

        request = {
            "request_id": request_id,
            "prompt": prompt,
        }
        request_fn = getattr(pyzlc, "call", None) or getattr(
            pyzlc, "zlc_request"
        )
        try:
            return request_fn(
                self._spatial_relation_service_name,
                request,
                timeout=self._spatial_relation_request_timeout,
                group_name=self._pyzlc_group_name,
            )
        except Exception as exc:
            return {
                "success": False,
                "request_id": request["request_id"],
                "message": str(exc),
            }

    def _spatial_relation_response_complete(self, response: Any) -> bool:
        if response is None:
            return True
        if not isinstance(response, dict):
            return True
        if response.get("scene_graph_complete"):
            return True
        if response.get("success") is False:
            return True
        return False

    def _print_spatial_relation_response(
        self,
        reason: str,
        response: Any,
    ) -> None:
        response_text = pprint.pformat(response)
        message = (
            f"{self._spatial_relation_service_name} {reason} response:\n"
            f"{response_text}"
        )
        pyzlc.info(message)

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

    def _infer_step(self) -> Optional[bool]:
        if self._awaiting_reset_confirmation or self._awaiting_eval_confirmation:
            self._maybe_request_spatial_relation()
            time.sleep(0.05)
            return False
        return super()._infer_step()

    def _start_reset_policy(self) -> None:
        pyzlc.sleep(2.0)  # brief pause before starting reset policy
        self._awaiting_reset_confirmation = False
        self._stop_infering()
        self._reset_arm()
        self._next_start_policy_name = "reset"
        self._start_infering()

    def _start_eval_policy(self) -> None:
        self._awaiting_eval_confirmation = False
        pyzlc.sleep(2.0)  # brief pause before starting eval policy
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
