from pathlib import Path
from typing import Any, List, Optional
import contextlib
import time
from unittest import result
import uuid
from httpcore import request
import pyzlc
import open3d as o3d
import pyzlc
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from ..control_pair.motion_planner_policy_panda_control_pair import (
    PolicyMotionPlannerControlPair,
)
from ..data_collection.irl_wrapper import IRLDataWrapper, ImageDataWrapper, PandaArmDataWrapper, RobotiqGripperDataWrapper
from .policy_inference_manager import PolicyInferenceManager
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose


GRASPNET_CHECKPOINT_PATH = (
    "/home/jjiang/jing/graspnet-baseline/model/checkpoint-rs.tar"
)
GRASPNET_NUM_POINT = 20000
GOAL_PCD_GRASPNET_NUM_POINT = 2048
GRASPNET_NUM_VIEW = 300
TOP_DOWN_GRASP_COS_THRESH = 0.9
POINT_CLOUD_STAT_NB_NEIGHBORS = 20
POINT_CLOUD_STAT_STD_RATIO = 2.0
POINT_CLOUD_CLUSTER_EPS_M = 0.02
POINT_CLOUD_CLUSTER_MIN_POINTS = 50
CameraImage = dict[str, Any]


def quat_xyzw_to_wxyz(quat: List[float]) -> List[float]:
    if len(quat) != 4:
        raise ValueError(f"Expected quaternion size 4, got {len(quat)}")
    return [quat[3], quat[0], quat[1], quat[2]]


def graspnet_rotation_to_curobo_tool_rotation(
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    tool_flip = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    return rotation_matrix @ tool_flip


def grasp_to_curobo_goal_tensors(
    grasp: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.as_tensor(
        grasp.translation,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, 1, 1, 3)

    tool_rotation = graspnet_rotation_to_curobo_tool_rotation(
        grasp.rotation_matrix
    )
    quat_xyzw = Rotation.from_matrix(tool_rotation).as_quat()
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw.tolist())
    quaternion = torch.as_tensor(
        quat_wxyz,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, 1, 1, 4)
    return position, quaternion


def vis_grasp(cloud: o3d.geometry.PointCloud, gg: GraspGroup) -> None:
    gg = gg.nms()
    gg.sort_by_score()
    gg = gg[:10]
    # pyzlc.info(f"Top grasps:\n{gg}")
    grippers = gg.to_open3d_geometry_list()
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,
        origin=np.array([0.0, 0.0, 0.0]),
    )
    o3d.visualization.draw_geometries([cloud, base_frame, *grippers])
    #fused point cloud visulization
    # o3d.visualization.draw_geometries([cloud, base_frame])


def filter_top_down_grasps(
    gg: GraspGroup,
    cos_thresh: float = TOP_DOWN_GRASP_COS_THRESH,
) -> GraspGroup:
    if len(gg) == 0:
        return gg

    top_down = np.array([0.0, 0.0, 1.0])
    approach = gg.rotation_matrices[:, :, 0]
    approach = approach / (np.linalg.norm(approach, axis=1, keepdims=True) + 1e-8)

    scores = np.dot(approach, top_down)
    pyzlc.info(
        f"Top-down filter score range: min={scores.min():.3f}, "
        f"max={scores.max():.3f}, threshold={cos_thresh:.3f}"
    )
    return gg[scores > cos_thresh]


def log_grasp_orientation_debug(gg: GraspGroup) -> None:
    if len(gg) == 0:
        return

    rotation_matrix = gg[0].rotation_matrix
    rotation = Rotation.from_matrix(rotation_matrix)
    pyzlc.info(
        "Top grasp orientation debug:\n"
        f"rotation matrix:\n{rotation_matrix}\n"
        f"euler xyz rad: {rotation.as_euler('xyz', degrees=False)}\n"
        f"euler xyz deg: {rotation.as_euler('xyz', degrees=True)}\n"
        f"euler zyx rad: {rotation.as_euler('zyx', degrees=False)}\n"
        f"euler zyx deg: {rotation.as_euler('zyx', degrees=True)}"
    )


class MotionPlannerInference(PolicyInferenceManager):
    def __init__(
        self,
        data_collectors: List[IRLDataWrapper],
        control_pair: PolicyMotionPlannerControlPair,
        task: str,
        cfg=None,
    ) -> None:
        super().__init__(
            task=task,
            fps=30 if cfg is None else cfg.fps,
            single_step=True,
        )
        self.data_collectors = data_collectors
        self.control_pair = control_pair
        for hw in data_collectors:
            if isinstance(hw, ImageDataWrapper) or hw.hw_type == "camera":
                self.cameras.append(hw)  # type: ignore[arg-type]
            elif isinstance(hw, PandaArmDataWrapper):
                self.follower_arm = hw.arm
            elif isinstance(hw, RobotiqGripperDataWrapper):
                self.gripper = hw.gripper
        self.running = True
        self.net = self._load_graspnet()
        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)
        self.grasp_config = MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model="/home/jjiang/jing/reset_control_client/configs/table_scene.yaml",
        )
        self.motion_planner = MotionPlanner(self.grasp_config)
        self.motion_planner.warmup(enable_graph=False, num_warmup_iterations=5)

    def _load_graspnet(self) -> GraspNet:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = GraspNet(
            input_feature_dim=0,
            num_view=GRASPNET_NUM_VIEW,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(device=device, dtype=torch.float32)
        checkpoint = torch.load(
            str(GRASPNET_CHECKPOINT_PATH),
            map_location=device,
        )
        net.load_state_dict(checkpoint["model_state_dict"])
        net.float()
        net.eval()
        pyzlc.info(
            f"Loaded GraspNet checkpoint {GRASPNET_CHECKPOINT_PATH} "
            f"(epoch: {checkpoint['epoch']})"
        )
        return net

    def _build_graspnet_end_points(
        self,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        workspace_mask: np.ndarray,
        cam_name: str,
    ) -> tuple[dict[str, Any], o3d.geometry.PointCloud]:

        color_sampled = color_masked[idxs]

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(
            cloud_masked.astype(np.float32)
        )
        cloud_o3d.colors = o3d.utility.Vector3dVector(
            color_masked.astype(np.float32)
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = torch.as_tensor(
            cloud_sampled[np.newaxis],
            dtype=torch.float32,
            device=device,
        ).contiguous()

        end_points = {
            "point_clouds": cloud_sampled,
            "cloud_colors": color_sampled,
        }
        return end_points, cloud_o3d

    def _build_graspnet_end_points_from_pcd_bytes(
        self,
        pcd_data: bytes,
    ) -> tuple[dict[str, Any], o3d.geometry.PointCloud]:
        cloud = np.frombuffer(pcd_data, dtype=np.float32).reshape(-1, 3)
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud) == 0:
            raise ValueError("Received empty goal point cloud.")

        target_num_points = GOAL_PCD_GRASPNET_NUM_POINT
        if len(cloud) >= target_num_points:
            idxs = np.random.choice(
                len(cloud),
                target_num_points,
                replace=False,
            )
        else:
            idxs1 = np.arange(len(cloud))
            idxs2 = np.random.choice(
                len(cloud),
                target_num_points - len(cloud),
                replace=True,
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud[idxs]

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = torch.as_tensor(
            cloud_sampled[np.newaxis],
            dtype=torch.float32,
            device=device,
        ).contiguous()

        return {"point_clouds": cloud_sampled}, cloud_o3d

    def _predict_grasps(
        self,
        net: GraspNet,
        end_points: dict[str, Any],
    ) -> GraspGroup:
        device = next(net.parameters()).device
        net.float()
        end_points["point_clouds"] = end_points["point_clouds"].to(
            device=device,
            dtype=torch.float32,
        ).contiguous()
        autocast_context = (
            torch.cuda.amp.autocast(enabled=False)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            with torch.no_grad(), autocast_context:
                end_points = net(end_points)
                grasp_pred = pred_decode(end_points)
        except RuntimeError as exc:
            point_clouds = end_points["point_clouds"]
            pyzlc.error(
                "GraspNet forward failed with "
                f"point_clouds dtype={point_clouds.dtype}, "
                f"device={point_clouds.device}, "
                f"model dtype={next(net.parameters()).dtype}"
            )
            raise
        gg_array = grasp_pred[0].detach().cpu().numpy()
        return GraspGroup(gg_array)

    def _generate_grasp_motion_plan(self, target_grasp: Any):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        position, quaternion = grasp_to_curobo_goal_tensors(target_grasp, device)
#         quaternion = torch.tensor(
#             [[[[[
#  -0.0077915681409994955, 0.9239915485916731, -0.3818021336541049, -0.02015044253208927,
#             ]]]]],
#             device=device,
#             dtype=torch.float32,
#         )
        position[..., 2] = 0.15 #0.067 lift the grasp pose up by 6cm to avoid collision during approach
        pyzlc.info(
            "Generated cuRobo grasp goal pose from GraspNet:\n"
            f"position={position.flatten().tolist()}\n"
            f"quaternion_wxyz={quaternion.flatten().tolist()}"
        )
        grasp_pose = GoalToolPose(
            tool_frames=self.motion_planner.tool_frames,
            position=position,
            quaternion=quaternion,
        )

        joint_names = self.motion_planner.joint_names
        state = self.follower_arm.get_franka_arm_state()
        q = torch.tensor([state["q"]], device=device, dtype=torch.float32)
        q_start = JointState.from_position(q, joint_names=joint_names)
        plan_result = self.motion_planner.plan_grasp(
            current_state=q_start,
            grasp_poses=grasp_pose,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
            grasp_lift_offset=0.15,
            grasp_lift_in_tool_frame=False,
        )
        approach = None
        grasp = None
        lift = None
        if plan_result.success is not None and plan_result.success.any():
            print("✓ Grasp planning succeeded!")

            approach = plan_result.approach_interpolated_trajectory
            grasp = plan_result.grasp_interpolated_trajectory
            lift = plan_result.lift_interpolated_trajectory

            if approach is not None:
                print(f"  Approach: {approach.position.shape[-2]} waypoints")
            if grasp is not None:
                print(f"  Grasp:    {grasp.position.shape[-2]} waypoints")
            if lift is not None:
                print(f"  Lift:     {lift.position.shape[-2]} waypoints")
        else:
            pyzlc.warning(f"Grasp planning failed: {plan_result.status}")
        return approach, grasp, lift
    
    def _infer_step(self) -> bool:
        start_time = time.perf_counter()
        request_id = uuid.uuid4().hex
        request = {
            "prompt": self.task,
            "request_id": request_id,
            "goal_key": self.task,
        }
        pyzlc.info(f"Sending request: {request}")
        service_name = "scene_graph"
        group_name = "robot_lab_robotiq_202"

        response = None
        while response is None:
            if not pyzlc.wait_for_service(
                service_name,
                timeout=5,
                group_name=group_name,
            ):
                pyzlc.warning(
                    f"Waiting for service {service_name!r} to become available."
                )
                time.sleep(0.5)
                continue

            response = pyzlc.call(
                service_name,
                request,
                timeout=300,
                group_name=group_name,
            )
            if response is None:
                pyzlc.warning(
                    f"Still waiting for service {service_name!r} response."
                )
                continue

            if response.get("request_id") != request_id:
                pyzlc.info(
                    f"Ignoring response for stale request: {response!r}"
                )
                response = None
                time.sleep(0.5)
                continue

            goal_pcd = response.get("goal_point_cloud", "")
            num_points = response.get("num_points", 0)
            if not goal_pcd:
                pyzlc.info(
                    f"Service {service_name!r} returned no goal point cloud: {response!r}"
                )
                response = None
                time.sleep(0.5)
                continue
            pyzlc.info(f"Received goal point cloud with {num_points} points")

        end_points, cloud = self._build_graspnet_end_points_from_pcd_bytes(goal_pcd)
        grasps = self._predict_grasps(self.net, end_points)
        num_grasps_before_filter = len(grasps)
        grasps = filter_top_down_grasps(grasps)
        grasps = grasps.sort_by_score()

        pyzlc.info(
            f"Top-down grasp filter kept {len(grasps)} / "
            f"{num_grasps_before_filter} grasps"
        )
        if len(grasps) == 0:
            pyzlc.warning("No top-down grasps found.")
            return False
        pyzlc.info(f"top top-down grasp:\n{grasps[0]}")
        # log_grasp_orientation_debug(grasps)
        # vis_grasp(cloud, grasps)

        approach, grasp, lift = self._generate_grasp_motion_plan(grasps[0])
        self.control_pair.send_joint_state_plan(
            approach,
            grasp,
            command_interval_s=0.1,
            gripper_cmd=None,
            max_waypoints_per_phase=5000,
        )
        self.control_pair.send_gripper_state(1, blocking=True)
        pyzlc.sleep(2.0)  # wait for gripper to close before lifting
        pyzlc.info("Gripper closed, starting lift motion.")
        self.control_pair.send_joint_state_plan(
            lift,
            command_interval_s=0.1,
            set_control_mode=False,
            gripper_cmd=1,
            max_waypoints_per_phase=5000,
        )

        elapsed = time.perf_counter() - start_time
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)
        return True

    def _close(self):
        self.running = False
        self.control_pair.stop_control_pair()
        return super()._close()

    def _reset_arm(self):
        self.control_pair.reset_action()
        self._ui_console.log("Resetting robot arm position...")
        try:
            self.control_pair.go_home()
            time.sleep(3)
            self._ui_console.log("Robot arm reset to home position.")
        except Exception as exc:
            self._ui_console.log(f"Failed to reset arm: {exc}")

    def _start_infering(self) -> None:
        self.control_pair.reset_action()
        super()._start_infering()

    def _save_episode(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode saved.")

    def _discard_infering(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode discarded.")

    def _stop_infering(self) -> None:
        super()._stop_infering()
