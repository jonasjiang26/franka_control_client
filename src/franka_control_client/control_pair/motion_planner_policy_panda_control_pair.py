from __future__ import annotations

import time
import traceback
from typing import Any, Optional
from scipy.spatial.transform import Rotation as R
import numpy as np
import pyzlc

from ..control_pair.policy_panda_control_pair import PolicyPandaControlPair
from ..franka_robot.panda_arm import ControlMode, RemotePandaArm
from ..robotiq_gripper.robotiq_gripper import RemoteRobotiqGripper
from .action_chunking_buffer import DeltaActionChunkingBuffer

DEFAULT_CONTROL_HZ: float = 500
GRIPPER_DEADBAND: float = 1e-3
GRIPPER_SPEED = 0.7
GRIPPER_FORCE = 0.3
ACTION_LOG_INTERVAL_S: float = 0.5
GRIPPER_TOGGLE_WARN_WINDOW_S: float = 3.0
GRIPPER_TOGGLE_WARN_COUNT: int = 6
DEFAULT_POSITION = (0.0, 0.0, 0.0, -2.15, 0.0, 2.15, 0.0)

# Calculate velocity limits using the standard approach from training
VELOCITY_LIMITS = np.array([[-4 * np.pi / 2, 4 * np.pi / 2]] * 7).T / 32
VELOCITY_LIMITS_NORM = np.linalg.norm(VELOCITY_LIMITS)


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"Expected quaternion size 4, got {quat.size}")
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


class PolicyMotionPlannerControlPair(PolicyPandaControlPair):
    """
    Apply policy actions to a Panda arm with a gripper.

    Action semantics: [x, y, z, qw, qx, qy, qz, gripper] (cuRobo cartesian pose + gripper).
    The quaternion is converted to Panda's [qx, qy, qz, qw] order before sending.
    Gripper value is normalized in [0, 1]. It is scaled to device range.
    Includes velocity and acceleration limiting for safety.
    """

    def __init__(
        self,
        panda_arm: RemotePandaArm,
        gripper: RemoteRobotiqGripper,
        control_hz: float = DEFAULT_CONTROL_HZ,
        action_chunk_size: int = 10,
        action_chunk_dt: float = 0.05,
    ) -> None:
        super().__init__(panda_arm, gripper, control_hz)
        # self.panda_arm = panda_arm
        # self.gripper = gripper
        # self.control_hz = float(control_hz)
        # self._action_lock = (
        #     threading.Lock()
        # )  # only one of the update_action and control_step visit latest_action at the same time
        self.action_buffer = DeltaActionChunkingBuffer(
            robot_arm=panda_arm,
            action_dt=action_chunk_dt,
            chunk_size=action_chunk_size,
            action_dim=8,
        )

        # Velocity limiting state
        self._last_joint_pos: Optional[np.ndarray] = None
        self._last_control_time: Optional[float] = None
        self._dt = 1.0 / self.control_hz  # time delta between control steps

    def get_lastest_command(self) -> Optional[np.ndarray]:
        return self._get_latest_action_from_chunk()

    def clear_lastest_command(self) -> None:
        self.action_buffer.clear()

    def _joint_state_commands_to_numpy(
        self,
        joint_state: Any,
    ) -> np.ndarray:
        commands = getattr(joint_state, "position", joint_state)
        if hasattr(commands, "detach"):
            commands = commands.detach().cpu().numpy()
        commands = np.asarray(commands, dtype=np.float64)

        if commands.ndim == 1:
            commands = commands.reshape(1, -1)
        elif commands.ndim > 2:
            # cuRobo trajectories can be shaped like (batch, seed, horizon, dim),
            # e.g. (1, 1, 5000, 9). Keep the last dimension as the command dim.
            commands = commands.reshape(-1, commands.shape[-1])

        if commands.shape[-1] < 7:
            raise ValueError(
                f"Expected command dimension >= 7, got {commands.shape}"
            )
        return commands

    def send_joint_state_plan(
        self,
        *joint_states: Any,
        command_interval_s: float = 0.02,
        set_control_mode: bool = True,
        gripper_cmd: Optional[float] = None,
        max_waypoints_per_phase: Optional[int] = 100,
        stop_when_reached: bool = True,
        joint_goal_tolerance: float = 0.1
    ) -> int:
        """
        Send every waypoint in one or more planned JointState trajectories.

        Args:
            *joint_states: JointState trajectories, tensors, or arrays. None values are skipped.
            command_interval_s: Sleep time between consecutive joint commands.
            set_control_mode: If True, switch the arm to HybridJointImpedance first.
            gripper_cmd: Optional gripper state to send with each waypoint.
            max_waypoints_per_phase: If set, uniformly downsample each phase to this many points.
            stop_when_reached: Stop a phase early if current joints are close to its final target.
            joint_goal_tolerance: L2 joint distance threshold for early stopping.

        Returns:
            Number of joint position commands sent.
        """
        if command_interval_s < 0.0:
            raise ValueError("command_interval_s must be non-negative")

        if set_control_mode:
            self.panda_arm.set_franka_arm_control_mode(
                ControlMode.HybridJointImpedance
            )

        num_sent = 0
        for joint_state in joint_states:
            if joint_state is None:
                continue

            commands = self._joint_state_commands_to_numpy(joint_state)
            original_count = len(commands)
            if (
                max_waypoints_per_phase is not None
                and original_count > max_waypoints_per_phase
            ):
                if max_waypoints_per_phase < 2:
                    raise ValueError("max_waypoints_per_phase must be at least 2")
                waypoint_idxs = np.linspace(
                    0,
                    original_count - 1,
                    max_waypoints_per_phase,
                    dtype=np.int64,
                )
                commands = commands[waypoint_idxs]
            pyzlc.info(
                f"Sending joint trajectory with {len(commands)} / "
                f"{original_count} waypoints "
                f"at interval {command_interval_s:.3f}s"
            )
            phase_goal = commands[-1, :7]
            for command in commands:
                joint_pos = command[:7]
                self.panda_arm.send_joint_position_command(joint_pos)
                self._last_joint_pos = np.asarray(joint_pos, dtype=np.float32)

                command_gripper = gripper_cmd
                if command_gripper is None and command.size >= 8:
                    command_gripper = float(command[7])
                if command_gripper is not None:
                    self.send_gripper_state(command_gripper)
                pyzlc.info(f"num_sent: {num_sent}")
                num_sent += 1
                if command_interval_s > 0.0:
                    pyzlc.sleep(command_interval_s)
                if stop_when_reached:
                    current_joint_pos = self._get_current_joint_pos()
                    if current_joint_pos is not None:
                        joint_error = np.linalg.norm(current_joint_pos - phase_goal)
                        if joint_error <= joint_goal_tolerance:
                            pyzlc.info(
                                f"Reached phase goal early after {num_sent} commands "
                                f"(joint error={joint_error:.4f})"
                            )
                            break

        pyzlc.info(f"Finished sending {num_sent} joint position commands")
        return num_sent

    def send_gripper_state(
        self,
        gripper_cmd: float,
        blocking: bool = False,
    ) -> None:
        gripper_cmd = 1 if float(gripper_cmd) >= 0.5 else 0.2
        if (
            self._last_gripper_cmd is None
            or abs(gripper_cmd - self._last_gripper_cmd) > GRIPPER_DEADBAND
        ):
            pyzlc.info(f"Sending gripper command: {gripper_cmd}")
            self.gripper.send_grasp_command(
                position=gripper_cmd,
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
                blocking=blocking,
            )
            self._last_gripper_cmd = gripper_cmd

    # using by policy side to update the latest action, and control loop will read the latest action and execute it
    def update_action(self, action: np.ndarray) -> None:
        """Update the latest action used by the control loop."""
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size == 7:
            absolute_action = self.action_buffer.delta2absolute(
                arr.reshape(1, -1)
            )[0]
        elif arr.size >= 8:
            absolute_action = arr[:8].astype(np.float32, copy=False)

        else:
            raise ValueError(
                f"Expected delta-cartesian action size 7 or absolute action size >= 8, got {arr.size}"
            )
        with self._action_lock:
            self._latest_action = np.array(absolute_action, copy=True)

    # using by policy side to update the latest action_chunk, and control loop will read the latest action and execute it
    def update_action_chunk(self, action_chunk: np.ndarray) -> None:
        """Update the latest action chunk used by the control loop."""
        self.action_buffer.add_new_action_chunk(action_chunk)

    def _get_latest_action(self) -> Optional[np.ndarray]:
        with self._action_lock:
            if self._latest_action is None:
                return None
            return self._latest_action.copy()

    def _get_latest_action_from_chunk(self) -> Optional[np.ndarray]:
        return self.action_buffer.get_action()

    def reset_action(self) -> None:
        """Reset the latest action state when starting a new episode."""
        with self._action_lock:
            self._latest_action = None
            self.action_buffer.clear()
        self.clear_lastest_command()
        self._last_gripper_cmd = None
        self._last_gripper_binary = None
        self._gripper_toggle_count = 0
        self._gripper_toggle_window_start_ts = time.time()
        self._last_cartesian_pos = self._get_current_cartesian_pose()
        pyzlc.info("Action state reset for new episode")

    def control_step(self) -> None:
        action = self.action_buffer.apply_action()
        if action is None:
            action = self._get_latest_action()
        if action is None:
            return
        self.panda_arm.send_cartesian_pose_command(
            action[:3],
            quat_wxyz_to_xyzw(action[3:7]),
        )
        
        print(f"Applied command: {action}")
        
        # Gripper command
        gripper_cmd = float(action[-1])
        gripper_cmd = 1 if gripper_cmd >= 0.5 else 0.2
        action[-1] = gripper_cmd
        if (
            self._last_gripper_cmd is None
            or abs(gripper_cmd - self._last_gripper_cmd) > GRIPPER_DEADBAND
        ):
            self.gripper.send_grasp_command(
                position=gripper_cmd,
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
                blocking=False,
            )
            self._last_gripper_cmd = gripper_cmd
        # End_time = time.perf_counter()
        # print(f"command took {End_time - start_time:.3f} seconds")

    def control_reset(self) -> None:
        self.panda_arm.set_franka_arm_control_mode(ControlMode.CartesianImpedance)
        current_pose = self._get_current_cartesian_pose()
        self.panda_arm.send_cartesian_pose_command(current_pose[:3], current_pose[3:])

    def control_end(self) -> None:
        self.panda_arm.set_franka_arm_control_mode(ControlMode.IDLE)

    def _control_task(self) -> None:
        try:
            self.control_reset()
            while self.is_running:
                start = time.perf_counter()
                self.control_step()
                if time.perf_counter() - start < (1.0 / self.control_hz):
                    pyzlc.sleep(
                        (1.0 / self.control_hz) - (time.perf_counter() - start)
                    )
            self.control_end()
        except Exception as e:
            print(f"Control task encountered an error: {e}")
            traceback.print_exc()

    def _get_current_cartesian_pose(self) -> np.ndarray:
        current_state = self.panda_arm.current_state
        if current_state is None or "EE_pos" not in current_state:
            raise ValueError(
                "Current arm state is not available or missing end-effector position"
            )
        cartesian_pos = np.asarray(
            current_state["EE_pos"], dtype=np.float32
        ).reshape(-1)
        cartesian_rot = np.asarray(
            current_state["EE_quat"], dtype=np.float32
        ).reshape(-1)
        cartesian_pose = np.concatenate([cartesian_pos, cartesian_rot])
        if cartesian_pose.size != 7:
            pyzlc.error(
                f"Unexpected current arm state size during control init: {cartesian_pose.size}"
            )
            raise ValueError("Unexpected current arm state size")
        return cartesian_pose
