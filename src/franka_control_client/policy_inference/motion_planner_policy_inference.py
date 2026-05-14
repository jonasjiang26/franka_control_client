from pathlib import Path
from typing import Any, List, Optional
import contextlib
import time
import uuid
from httpcore import request
import pyzlc
import open3d as o3d
import pyzlc
import numpy as np
import torch
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from ..control_pair.motion_planner_policy_panda_control_pair import (
    PolicyMotionPlannerControlPair,
)
from ..data_collection.irl_wrapper import IRLDataWrapper, ImageDataWrapper
from .policy_inference_manager import PolicyInferenceManager


GRASPNET_CHECKPOINT_PATH = (
    "/home/jjiang/jing/graspnet-baseline/model/checkpoint-rs.tar"
)
GRASPNET_NUM_POINT = 20000
GRASPNET_NUM_VIEW = 300
POINT_CLOUD_STAT_NB_NEIGHBORS = 20
POINT_CLOUD_STAT_STD_RATIO = 2.0
POINT_CLOUD_CLUSTER_EPS_M = 0.02
POINT_CLOUD_CLUSTER_MIN_POINTS = 50
CameraImage = dict[str, Any]


def vis_grasp(cloud: o3d.geometry.PointCloud, gg: GraspGroup) -> None:
    gg.nms()
    gg.sort_by_score()
    gg = gg[:20]
    # pyzlc.info(f"Top grasps:\n{gg}")
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


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

        self.running = True
        self.net = self._load_graspnet()
        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

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

        if len(cloud) >= GRASPNET_NUM_POINT:
            idxs = np.random.choice(
                len(cloud),
                GRASPNET_NUM_POINT,
                replace=False,
            )
        else:
            idxs1 = np.arange(len(cloud))
            idxs2 = np.random.choice(
                len(cloud),
                GRASPNET_NUM_POINT - len(cloud),
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


    def _process_camera_frame(
        self,
        cam_name: str,
        cam_image: CameraImage,
    ) -> None:
        rgb, depth = self._decode_camera_image(cam_image)
        segmentation_mask = self._resize_mask_to_depth(
            self._segment_rgb(rgb),
            depth,
        )
        segmented_depth = np.where(segmentation_mask, depth, 0).astype(np.uint16)

        self._project_and_store_segmented_depth(
            cam_name,
            segmented_depth,
            segmentation_mask,
        )

        end_points, cloud = self._build_graspnet_end_points(
            segmented_depth,
            rgb,
            segmentation_mask,
            cam_name,
        )
        grasps = self._predict_grasps(self.net, end_points)
        vis_grasp(cloud, grasps)

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
        vis_grasp(cloud, grasps)

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
