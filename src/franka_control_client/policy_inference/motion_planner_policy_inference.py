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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from graspnet import GraspNet, pred_decode
from graspnetAPI import Grasp, GraspGroup
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
GRASPNET_NUM_POINT = 30000
GOAL_PCD_GRASPNET_NUM_POINT = GRASPNET_NUM_POINT
GOAL_PCD_UPSAMPLE_K_NEIGHBORS = 6
GRASPNET_NUM_VIEW = 300
TOP_DOWN_GRASP_COS_THRESH = 0.1
CENTER_GRASP_MAX_XY_DISTANCE_M = 0.06
MIN_GRASP_WIDTH_M = 0.03
MAX_GRASP_WIDTH_M = 0.055
POINT_CLOUD_STAT_NB_NEIGHBORS = 20
POINT_CLOUD_STAT_STD_RATIO = 2.0
POINT_CLOUD_CLUSTER_EPS_M = 0.02
POINT_CLOUD_CLUSTER_MIN_POINTS = 50
CameraImage = dict[str, Any]
TOP_DOWN_QUAT_WXYZ = [
    0.9996659647740053,
    -0.006917926433006643,
    -0.0233896430534402,
    -0.00854551134557857,
]
GRASP_YAW_AXIS = 2
GRASP_MOTION_PLAN_YAW_OFFSET_DEG = 45.0
INITIAL_POSITION_XYZ = [
    0.4569405479296274,
    0.1208722997215983,
    0.18180993839463131,
]
INITIAL_QUAT_XYZW = [
    0.9200004670002642,
    0.39068259597851607,
    0.02467542003362327,
    0.018904326619791186,
]


def upsample_point_cloud_by_interpolation(
    cloud: np.ndarray,
    target_num_points: int,
    k_neighbors: int = GOAL_PCD_UPSAMPLE_K_NEIGHBORS,
) -> np.ndarray:
    if len(cloud) >= target_num_points:
        return cloud.astype(np.float32, copy=False)
    if len(cloud) < 2:
        idxs = np.random.choice(len(cloud), target_num_points, replace=True)
        return cloud[idxs].astype(np.float32, copy=False)

    k = min(k_neighbors + 1, len(cloud))
    _, neighbor_idxs = cKDTree(cloud).query(cloud, k=k)
    neighbor_idxs = np.atleast_2d(neighbor_idxs)[:, 1:]
    if neighbor_idxs.shape[1] == 0:
        idxs = np.random.choice(len(cloud), target_num_points, replace=True)
        return cloud[idxs].astype(np.float32, copy=False)

    num_new_points = target_num_points - len(cloud)
    base_idxs = np.random.choice(len(cloud), num_new_points, replace=True)
    neighbor_cols = np.random.randint(0, neighbor_idxs.shape[1], size=num_new_points)
    paired_idxs = neighbor_idxs[base_idxs, neighbor_cols]
    alpha = np.random.uniform(0.15, 0.85, size=(num_new_points, 1)).astype(np.float32)
    new_points = cloud[base_idxs] * (1.0 - alpha) + cloud[paired_idxs] * alpha
    return np.concatenate([cloud, new_points], axis=0).astype(np.float32, copy=False)


def quat_xyzw_to_wxyz(quat: List[float]) -> List[float]:
    if len(quat) != 4:
        raise ValueError(f"Expected quaternion size 4, got {len(quat)}")
    return [quat[3], quat[0], quat[1], quat[2]]


def quat_wxyz_to_xyzw(quat: List[float]) -> List[float]:
    if len(quat) != 4:
        raise ValueError(f"Expected quaternion size 4, got {len(quat)}")
    return [quat[1], quat[2], quat[3], quat[0]]


def top_down_rotation_with_grasp_yaw(
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    tool_x = rotation_matrix[:, GRASP_YAW_AXIS].astype(np.float64)
    tool_x[2] = 0.0
    tool_x_norm = np.linalg.norm(tool_x)
    if tool_x_norm < 1e-6:
        tool_x = np.array([1.0, 0.0, 0.0])
    else:
        tool_x = tool_x / tool_x_norm

    tool_z = np.array([0.0, 0.0, -1.0])
    tool_y = np.cross(tool_z, tool_x)
    tool_y = tool_y / (np.linalg.norm(tool_y) + 1e-8)
    tool_x = np.cross(tool_y, tool_z)
    tool_x = tool_x / (np.linalg.norm(tool_x) + 1e-8)
    tool_rotation = np.column_stack([tool_x, tool_y, tool_z])
    ground_z_flip = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    return ground_z_flip @ tool_rotation


def grasp_to_curobo_goal_tensors(
    grasp: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.as_tensor(
        grasp.translation,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, 1, 1, 3)

    tool_rotation = top_down_rotation_with_grasp_yaw(
        grasp.rotation_matrix
    )
    yaw_offset = Rotation.from_euler(
        "z",
        GRASP_MOTION_PLAN_YAW_OFFSET_DEG,
        degrees=True,
    ).as_matrix()
    tool_rotation = yaw_offset @ tool_rotation
    quat_xyzw = Rotation.from_matrix(tool_rotation).as_quat()
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw.tolist())
    quaternion = torch.as_tensor(
        quat_wxyz,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, 1, 1, 4)
    return position, quaternion


def vis_grasp(cloud: o3d.geometry.PointCloud, grasps: Any) -> None:
    gripper_api_from_tool = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    if hasattr(grasps, "nms"):
        grasps = grasps.nms()
        grasps.sort_by_score()
        grasps = grasps[:10]
        grasps = GraspGroup(grasps.grasp_group_array.copy())
        if len(grasps) > 0:
            grasps.rotation_matrices = np.stack(
                [
                    top_down_rotation_with_grasp_yaw(rotation_matrix)
                    @ gripper_api_from_tool
                    for rotation_matrix in grasps.rotation_matrices
                ],
                axis=0,
            )
        grippers = grasps.to_open3d_geometry_list()
    else:
        grasps = Grasp(grasps.grasp_array.copy())
        grasps.rotation_matrix = top_down_rotation_with_grasp_yaw(
            grasps.rotation_matrix
        ) @ gripper_api_from_tool
        grippers = [grasps.to_open3d_geometry()]

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


def filter_grasps_near_point_cloud_center(
    gg: GraspGroup,
    cloud: o3d.geometry.PointCloud,
    max_xy_distance_m: float = CENTER_GRASP_MAX_XY_DISTANCE_M,
) -> GraspGroup:
    if len(gg) == 0:
        return gg

    points = np.asarray(cloud.points, dtype=np.float32)
    if len(points) == 0:
        return gg

    center_xy = points[:, :2].mean(axis=0)
    grasp_xy = gg.translations[:, :2]
    distances = np.linalg.norm(grasp_xy - center_xy, axis=1)
    pyzlc.info(
        f"Center grasp filter distance range: min={distances.min():.3f}m, "
        f"max={distances.max():.3f}m, threshold={max_xy_distance_m:.3f}m, "
        f"center_xy={center_xy.tolist()}"
    )
    return gg[distances <= max_xy_distance_m]


def filter_grasps_by_width(
    gg: GraspGroup,
    min_width_m: float = MIN_GRASP_WIDTH_M,
    max_width_m: float = MAX_GRASP_WIDTH_M,
) -> GraspGroup:
    if len(gg) == 0:
        return gg

    widths = gg.widths
    pyzlc.info(
        f"Grasp width range: min={widths.min():.3f}m, "
        f"max={widths.max():.3f}m, "
        f"keeping [{min_width_m:.3f}, {max_width_m:.3f}]m"
    )
    return gg[(widths >= min_width_m) & (widths <= max_width_m)]


def rank_grasps_by_distance_to_point_cloud_center(
    gg: GraspGroup,
    cloud: o3d.geometry.PointCloud,
) -> GraspGroup:
    if len(gg) == 0:
        return gg

    points = np.asarray(cloud.points, dtype=np.float32)
    if len(points) == 0:
        return gg

    center_xy = points[:, :2].mean(axis=0)
    distances = np.linalg.norm(gg.translations[:, :2] - center_xy, axis=1)
    order = np.argsort(distances)
    pyzlc.info(
        f"Closest grasp distance to point cloud center: "
        f"{distances[order[0]]:.3f}m"
    )
    return gg[order]


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
        scene: str,
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
        self.scene = scene

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
        source_num_points = len(cloud)
        dense_cloud = upsample_point_cloud_by_interpolation(cloud, target_num_points)
        if len(dense_cloud) != source_num_points:
            pyzlc.info(
                f"Upsampled goal point cloud from {source_num_points} to "
                f"{len(dense_cloud)} points for GraspNet and visualization"
            )

        if len(dense_cloud) >= target_num_points:
            idxs = np.random.choice(
                len(dense_cloud),
                target_num_points,
                replace=False,
            )
        else:
            idxs1 = np.arange(len(dense_cloud))
            idxs2 = np.random.choice(
                len(dense_cloud),
                target_num_points - len(dense_cloud),
                replace=True,
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = dense_cloud[idxs]

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(dense_cloud.astype(np.float64))

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
        position[..., 2] += 0.16 #= 0.17 lift the grasp pose up by 6cm to avoid collision during approach
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
            grasp_approach_offset=-0.15,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
            grasp_lift_offset=0.35,
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
    
    def generate_motion_plan_to_goal(
        self,
        position_xyz: List[float],
        quaternion_xyzw: List[float],
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        position = torch.as_tensor(
            position_xyz,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 3)
        quaternion = torch.as_tensor(
            quat_xyzw_to_wxyz(quaternion_xyzw),
            device=device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 4)
        goal_pose = GoalToolPose(
            tool_frames=self.motion_planner.tool_frames,
            position=position,
            quaternion=quaternion,
        )

        joint_names = self.motion_planner.joint_names
        state = self.follower_arm.get_franka_arm_state()
        q = torch.tensor([state["q"]], device=device, dtype=torch.float32)
        q_start = JointState.from_position(q, joint_names=joint_names)
        plan_result = self.motion_planner.plan_pose(goal_pose, q_start)
        if plan_result.success is not None and plan_result.success.any():
            pyzlc.info("✓ Motion planning to goal succeeded!")
            return plan_result.interpolated_trajectory
        else:
            pyzlc.warning("Motion planning to goal failed.")
            return None
    
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
        # vis_grasp(cloud, grasps[:20])

        num_grasps_before_filter = len(grasps)
        grasps = filter_top_down_grasps(grasps)
        num_grasps_after_top_down_filter = len(grasps)
        grasps = filter_grasps_by_width(grasps)
        num_grasps_after_width_filter = len(grasps)
        grasps = rank_grasps_by_distance_to_point_cloud_center(grasps, cloud)

        pyzlc.info(
            f"Top-down grasp filter kept {num_grasps_after_top_down_filter} / "
            f"{num_grasps_before_filter} grasps; "
            f"width filter kept {num_grasps_after_width_filter}; "
            f"ranked {len(grasps)} grasps by distance to point cloud center"
        )
        if len(grasps) == 0:
            pyzlc.warning("No top-down grasps found.")
            return False
        pyzlc.info(f"top top-down grasp:\n{grasps[0]}")
        # log_grasp_orientation_debug(grasps)
        # vis_grasp(cloud, grasps[0])

        approach, grasp, lift = self._generate_grasp_motion_plan(grasps[0])
        self.control_pair.send_joint_state_plan(
            approach,
            grasp,
            command_interval_s=0.1,
            gripper_cmd=None,
            max_waypoints_per_phase=5000,
            joint_goal_tolerance=0.15,
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
            joint_goal_tolerance=0.2
        )

        place = self.generate_motion_plan_to_goal(
            INITIAL_POSITION_XYZ,
            INITIAL_QUAT_XYZW,
        )
        self.control_pair.send_joint_state_plan(
            place,
            command_interval_s=0.1,
            set_control_mode=False,
            gripper_cmd=1,
            max_waypoints_per_phase=5000,
            joint_goal_tolerance=0.1
        )
        self.control_pair.send_gripper_state(0, blocking=True)
        pyzlc.info("Place motion sent and gripper opened.")
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)
        return True

    def _close(self):
        self.running = False
        self.control_pair.stop_control_pair()
        if hasattr(self, "motion_planner"):
            self.motion_planner.destroy()
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
