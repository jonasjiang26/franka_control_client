from pathlib import Path
from typing import Any, List, Optional
import contextlib
import time
import sys

import cv2
import open3d as o3d
import pyzlc
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from lang_sam import LangSAM
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
GRASPNET_BASELINE_DIR = REPO_ROOT / "graspnet-baseline"
GRASPNET_API_DIR = REPO_ROOT / "graspnetAPI"


def _add_graspnet_paths() -> None:
    for path in (
        GRASPNET_BASELINE_DIR,
        GRASPNET_BASELINE_DIR / "models",
        GRASPNET_BASELINE_DIR / "dataset",
        GRASPNET_BASELINE_DIR / "utils",
        GRASPNET_BASELINE_DIR / "pointnet2",
        GRASPNET_API_DIR,
    ):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_graspnet_paths()

from graspnet import GraspNet, pred_decode
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup
from ..control_pair.motion_planner_policy_panda_control_pair import (
    PolicyMotionPlannerControlPair,
)
from ..data_collection.irl_wrapper import IRLDataWrapper, ImageDataWrapper
from .lerobot_policy_inference import (
    LeRobotPolicyInferenceConfig,
)
from .policy_inference_manager import PolicyInferenceManager


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "camera" / "calibration"
GRASPNET_CHECKPOINT_PATH = (
    REPO_ROOT / "graspnet-baseline" / "model" / "checkpoint-rs.tar"
)
DEFAULT_DEPTH_SCALE_M = 0.001
MAX_PROJECTED_POINTS = 20000
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
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


class MotionPlannerInference(PolicyInferenceManager):
    def __init__(
        self,
        data_collectors: List[IRLDataWrapper],
        control_pair: PolicyMotionPlannerControlPair,
        task: str,
        cfg: Optional[LeRobotPolicyInferenceConfig] = None,
    ) -> None:
        super().__init__(
            task=task,
            fps=30 if cfg is None else cfg.fps,
            single_step=True,
        )
        self.data_collectors = data_collectors
        self.control_pair = control_pair
        self.cameras: List[ImageDataWrapper] = []
        for hw in data_collectors:
            if isinstance(hw, ImageDataWrapper) or hw.hw_type == "camera":
                self.cameras.append(hw)  # type: ignore[arg-type]

        self.running = True
        self.sam = LangSAM()
        self.net = self._load_graspnet()
        self.camera_intrinsics = self._load_camera_intrinsics()
        self.camera_extrinsics = self._load_camera_extrinsics()
        self.latest_segmented_points_base: dict[str, np.ndarray] = {}
        self.latest_segmented_centroids_base: dict[str, np.ndarray] = {}
        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}")
        return data

    def _load_intrinsics_file(self, path: Path) -> dict[str, float]:
        data = self._load_yaml(path)
        return {
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "depth_scale": float(data.get("depth_scale", DEFAULT_DEPTH_SCALE_M)),
        }

    def _load_camera_intrinsics(self) -> dict[str, dict[str, float]]:
        intrinsics: dict[str, dict[str, float]] = {}
        fallback_path = CALIBRATION_DIR / "cam_intrinsics.yaml"
        fallback_data = (
            self._load_intrinsics_file(fallback_path)
            if fallback_path.exists()
            else None
        )

        for cam in self.cameras:
            intrinsics_path = CALIBRATION_DIR / f"{cam.hw_name}_intrinsics.yaml"
            if intrinsics_path.exists():
                data = self._load_intrinsics_file(intrinsics_path)
            elif fallback_data is not None:
                data = fallback_data
                pyzlc.warning(
                    f"Using fallback intrinsics {fallback_path} for {cam.hw_name}; "
                    f"create {intrinsics_path.name} to override."
                )
            else:
                pyzlc.warning(f"No intrinsics found for {cam.hw_name}.")
                continue

            intrinsics[cam.hw_name] = dict(data)
        return intrinsics

    def _load_camera_extrinsics(self) -> dict[str, np.ndarray]:
        extrinsics: dict[str, np.ndarray] = {}

        eye_to_hand_path = CALIBRATION_DIR / "eye_to_hand.yaml"
        if eye_to_hand_path.exists():
            data = self._load_yaml(eye_to_hand_path)
            camera = str(data.get("camera", "static_cam"))
            extrinsics[camera] = np.asarray(
                data["transforms"]["T_base_camera"]["matrix"],
                dtype=np.float64,
            )

        wrist_path = CALIBRATION_DIR / "wrist_cam_hand_eye.yaml"
        if wrist_path.exists():
            data = self._load_yaml(wrist_path)
            camera = str(data.get("camera", "wrist_cam"))
            extrinsics[camera] = np.asarray(
                data["transforms"]["T_hand_camera"]["matrix"],
                dtype=np.float64,
            )

        return extrinsics

    def _current_T_base_hand(self) -> np.ndarray:
        pose = self.control_pair._get_current_cartesian_pose()
        if pose.size != 7:
            raise ValueError(f"Expected current Cartesian pose size 7, got {pose.size}")

        T_base_hand = np.eye(4, dtype=np.float64)
        T_base_hand[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        T_base_hand[:3, 3] = pose[:3]
        return T_base_hand

    def _T_base_camera(self, cam_name: str) -> Optional[np.ndarray]:
        T = self.camera_extrinsics.get(cam_name)
        if T is None:
            pyzlc.warning(f"No extrinsics found for {cam_name}.")
            return None

        if cam_name == "wrist_cam":
            return self._current_T_base_hand() @ T
        return T

    def _segmented_depth_to_base_points(
        self,
        cam_name: str,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        intrinsics = self.camera_intrinsics.get(cam_name)
        if intrinsics is None:
            pyzlc.warning(f"No intrinsics found for {cam_name}; cannot project depth.")
            return np.empty((0, 3), dtype=np.float64)

        T_base_camera = self._T_base_camera(cam_name)
        if T_base_camera is None:
            return np.empty((0, 3), dtype=np.float64)

        valid = mask & (depth > 0)
        v, u = np.nonzero(valid)
        if v.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if v.size > MAX_PROJECTED_POINTS:
            sample_idx = np.linspace(0, v.size - 1, MAX_PROJECTED_POINTS).astype(np.int64)
            v = v[sample_idx]
            u = u[sample_idx]

        z = depth[v, u].astype(np.float64) * intrinsics["depth_scale"]
        x = (u.astype(np.float64) - intrinsics["cx"]) * z / intrinsics["fx"]
        y = (v.astype(np.float64) - intrinsics["cy"]) * z / intrinsics["fy"]
        points_camera = np.column_stack((x, y, z, np.ones_like(z)))
        points_base = (T_base_camera @ points_camera.T).T[:, :3]
        return points_base

    def _project_and_store_segmented_depth(
        self,
        cam_name: str,
        segmented_depth: np.ndarray,
        segmentation_mask: np.ndarray,
    ) -> np.ndarray:
        points_base = self._segmented_depth_to_base_points(
            cam_name,
            segmented_depth,
            segmentation_mask,
        )
        self.latest_segmented_points_base[cam_name] = points_base
        if points_base.size > 0:
            centroid_base = points_base.mean(axis=0)
            self.latest_segmented_centroids_base[cam_name] = centroid_base
            pyzlc.info(
                f"{cam_name} segmented depth projected to base: "
                f"{points_base.shape[0]} points, centroid xyz={centroid_base}"
            )
        else:
            self.latest_segmented_centroids_base.pop(cam_name, None)
            pyzlc.warning(
                f"{cam_name} segmented depth projection produced no valid points."
            )
        return points_base

    def _build_images(self) -> dict[str, CameraImage]:
        images: dict[str, CameraImage] = {}
        for cam in self.cameras:
            rgb_image, depth_image = cam.capture_step()
            if rgb_image is None:
                continue
            if not isinstance(rgb_image, np.ndarray):
                raise ValueError(
                    f"Expected numpy RGB image for {cam.hw_name}, got {type(rgb_image)}"
                )

            h, w, c = rgb_image.shape
            if depth_image is not None and depth_image.shape != (h, w):
                depth_image = cv2.resize(
                    depth_image,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            images[cam.hw_name] = {
                "height": int(h),
                "width": int(w),
                "channels": int(c),
                "rgb_data": np.ascontiguousarray(rgb_image).tobytes(),
                "depth_data": None
                if depth_image is None
                else np.ascontiguousarray(
                    depth_image.astype(np.uint16, copy=False)
                ).tobytes(),
            }
        return images

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

    def _camera_info_for_graspnet(
        self,
        cam_name: str,
        width: int,
        height: int,
    ) -> CameraInfo:
        intrinsics = (
            self.camera_intrinsics.get(cam_name)
            or self.camera_intrinsics.get("right_cam")
            or self.camera_intrinsics.get("static_cam")
        )
        if intrinsics is None:
            raise ValueError(f"No camera intrinsics available for {cam_name}.")

        factor_depth = 1.0 / intrinsics["depth_scale"]
        return CameraInfo(
            width,
            height,
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
            factor_depth,
        )

    def _build_graspnet_end_points(
        self,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        workspace_mask: np.ndarray,
        cam_name: str,
    ) -> tuple[dict[str, Any], o3d.geometry.PointCloud]:
        height, width = depth_image.shape
        camera = self._camera_info_for_graspnet(cam_name, width, height)
        cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)

        mask = (workspace_mask & (depth_image > 0))
        cloud_masked = cloud[mask]
        color_masked = (rgb_image.astype(np.float32) / 255.0)[mask]
        if len(cloud_masked) == 0:
            raise ValueError("No valid segmented depth points for GraspNet.")

        cloud_masked, color_masked = self._filter_point_cloud_noise(
            cloud_masked,
            color_masked,
        )
        if len(cloud_masked) == 0:
            raise ValueError("Point cloud filter removed all segmented points.")

        if len(cloud_masked) >= GRASPNET_NUM_POINT:
            idxs = np.random.choice(
                len(cloud_masked),
                GRASPNET_NUM_POINT,
                replace=False,
            )
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked),
                GRASPNET_NUM_POINT - len(cloud_masked),
                replace=True,
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
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

    def _filter_point_cloud_noise(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(points) < POINT_CLOUD_CLUSTER_MIN_POINTS:
            return points, colors

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))

        filtered_cloud, inlier_indices = cloud.remove_statistical_outlier(
            nb_neighbors=POINT_CLOUD_STAT_NB_NEIGHBORS,
            std_ratio=POINT_CLOUD_STAT_STD_RATIO,
        )
        inlier_indices = np.asarray(inlier_indices, dtype=np.int64)
        if inlier_indices.size == 0:
            return points, colors

        filtered_points = points[inlier_indices]
        filtered_colors = colors[inlier_indices]
        if len(filtered_points) < POINT_CLOUD_CLUSTER_MIN_POINTS:
            return filtered_points, filtered_colors

        filtered_cloud.points = o3d.utility.Vector3dVector(
            filtered_points.astype(np.float32)
        )
        labels = np.asarray(
            filtered_cloud.cluster_dbscan(
                eps=POINT_CLOUD_CLUSTER_EPS_M,
                min_points=POINT_CLOUD_CLUSTER_MIN_POINTS,
                print_progress=False,
            )
        )
        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            return filtered_points, filtered_colors

        largest_label = np.bincount(valid_labels).argmax()
        keep = labels == largest_label
        pyzlc.info(
            f"Filtered segmented point cloud from {len(points)} to {int(keep.sum())} points."
        )
        return filtered_points[keep], filtered_colors[keep]

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

    def _decode_camera_image(
        self,
        cam_image: CameraImage,
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.frombuffer(cam_image["rgb_data"], dtype=np.uint8).reshape(
            cam_image["height"],
            cam_image["width"],
            cam_image["channels"],
        )

        depth_data = cam_image.get("depth_data")
        if depth_data is None:
            raise ValueError("No depth data available.")

        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
            cam_image["height"],
            cam_image["width"],
        )
        return rgb, depth

    def _segment_rgb(self, rgb: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(rgb).convert("RGB")
        results = self.sam.predict([image_pil], [self.task])
        segmentation_mask = results[0]["masks"]
        if segmentation_mask.ndim == 3:
            segmentation_mask = np.any(segmentation_mask, axis=0)
        return segmentation_mask.astype(bool)

    def _resize_mask_to_depth(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        if mask.shape == depth.shape:
            return mask
        return cv2.resize(
            mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

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

    def _infer_step(self) -> None:
        start_time = time.perf_counter()
        images = self._build_images()
        for cam in self.cameras:
            if cam.hw_name not in images:
                pyzlc.warning(f"No image data available for {cam.hw_name}.")
                continue
            self._process_camera_frame(cam.hw_name, images[cam.hw_name])

        elapsed = time.perf_counter() - start_time
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

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
