import threading
from typing import Protocol, Optional, List, Dict, Any
from ..camera.camera import CameraDevice
from ..franka_robot.franka_arm import RemoteFranka
from ..franka_robot.franka_gripper import RemoteFrankaGripper

import sys
import select
import tty
import termios
import numpy as np
from pathlib import Path
import time
import cv2
import json
from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature
from PIL import Image
import pyzlc
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

class UIConsole:
    """
    Manages terminal output by separating persistent logs from
    transient interactive hints using ANSI escape sequences.
    """

    def __init__(self):
        self._current_hint = ""
        self._lock = threading.Lock()

    def update_hint(self, message: str):
        """
        Updates the bottom interactive instruction.
        This will be overwritten by the next hint or pushed down by a log.
        """
        with self._lock:
            self._current_hint = message
            # \r: Carriage return (to start of line)
            # \033[K: Clear line from cursor to end
            sys.stdout.write(f"\r\033[K{self._current_hint}")
            sys.stdout.flush()

    def log(self, message: str):
        """
        Prints a persistent log message that scrolls upward.
        Automatically restores the current hint at the bottom.
        """
        with self._lock:
            # 1. Clear the current hint line
            sys.stdout.write("\r\033[K")
            # 2. Print the log message with a newline
            sys.stdout.write(f"{message}\n")
            # 3. Restore the hint at the bottom
            sys.stdout.write(self._current_hint)
            sys.stdout.flush()


class NonBlockingKeyPress(object):
    """
    This class was copied and adapted from: https://stackoverflow.com/a/10079805
    Note that this solution is sometimes confused when spamming a character and that there are problems with special characters such as arrow keys.
    """

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            # Read one character
            data = sys.stdin.read(1)
            # Flush received but not read and written but not transmitted data
            # This does no fully fix that the input is confused if a key is spammed
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            return data
        return False


class DataCollector(Protocol):

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def save_step(self) -> None: ...

    def save_episode(self) -> None: ...

    def reset(self) -> None: ...

    def discard(self) -> None: ...


class LeRobotDataCollector:
    """Concrete DataCollector that combines image and robot data in LeRobot format."""
    
    def __init__(
        self,
        camera_device: CameraDevice,
        robot_arm: RemoteFranka,
        robot_gripper: RemoteFrankaGripper,
        dataset_root: str = "LeRobotDataset",
        fps: int = 30,
        video_codec: str = 'mp4v',
        chunk_size: int = 1000
    ):
        # Create wrappers
        self.image_wrapper = ImageDataWrapper(camera_device, fps, video_codec)
        self.robot_wrapper = RobotDataWrapper(robot_arm, robot_gripper)
        
        # Episode tracking
        self.episode_index = 0
        self.global_index = 0
        self.chunk_size = chunk_size
        self.fps = fps
        self.video_codec = video_codec
        
        # LeRobot dataset structure
        self.dataset_root = Path(dataset_root)
        self.data_dir = self.dataset_root / "data"
        self.videos_dir = self.dataset_root / "videos"
        self.meta_dir = self.dataset_root / "meta"
        
        # Create directories
        for dir_path in [self.data_dir, self.videos_dir, self.meta_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self) -> None:
        """Initialize metadata files if they don't exist."""
        info_path = self.meta_dir / "info.json"
        if not info_path.exists():
            info = {
                "codebase_version": "v2.0",
                "fps": self.fps,
                "video_codec": self.video_codec,
                "features": {
                    "observation.image": {
                        "dtype": "video",
                        "shape": [3, 480, 640],
                        "names": ["channel", "height", "width"]
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [-1],
                        "names": None
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [-1],
                        "names": None
                    },
                    "timestamp": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": None
                    },
                    "next.reward": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": None
                    },
                    "next.done": {
                        "dtype": "bool",
                        "shape": [1],
                        "names": None
                    },
                    "next.success": {
                        "dtype": "bool",
                        "shape": [1],
                        "names": None
                    },
                    "episode_index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    },
                    "frame_index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    },
                    "index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    },
                    "task_index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    }
                }
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
    
    def _get_chunk_dir(self, episode_idx: int) -> str:
        """Get chunk directory name based on episode index."""
        chunk_idx = episode_idx // self.chunk_size
        return f"chunk-{chunk_idx:03d}"
    
    def save_step(
        self,
        action: List[float],
        observation_state: Optional[List[float]] = None,
        reward: float = 0.0,
        done: bool = False,
        success: bool = False,
        task_index: int = 0,
    ) -> None:
        """Save step data to both image and robot wrappers."""
        # Save image frame
        self.image_wrapper.save_step()
        
        # Save robot state
        # self.robot_wrapper.save_step(
        #     action=action,
        #     observation_state=observation_state,
        #     reward=reward,
        #     done=done,
        #     success=success,
        #     task_index=task_index
        # )
    
    def save_episode(self) -> None:
        """Combine image and robot data, save in LeRobot format."""
        frames = self.image_wrapper.get_frames()
        # robot_data = self.robot_wrapper.get_data()
        
        # if not frames or not robot_data:
        #     print("No data to save")
        #     return
        
        # if len(frames) != len(robot_data):
        #     print(f"Warning: Frame count ({len(frames)}) != robot data count ({len(robot_data)})")
        #     min_len = min(len(frames), len(robot_data))
        #     frames = frames[:min_len]
        #     robot_data = robot_data[:min_len]
        
        try:
            chunk_dir = self._get_chunk_dir(self.episode_index)
            
            # Create chunk directories
            data_chunk_dir = self.data_dir / chunk_dir
            video_chunk_dir = self.videos_dir / chunk_dir / "observation.image"
            data_chunk_dir.mkdir(parents=True, exist_ok=True)
            video_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Save video file
            video_path = video_chunk_dir / f"episode_{self.episode_index:06d}.mp4"
            self._save_video(video_path, frames)
            
            # Add episode_index and global index to robot data
            combined_data = []
            # for step in robot_data:
            #     step['episode_index'] = self.episode_index
            #     step['index'] = self.global_index
            #     combined_data.append(step)
            #     self.global_index += 1
            
            # Save parquet file with numerical data
            parquet_path = data_chunk_dir / f"episode_{self.episode_index:06d}.parquet"
            self._save_parquet(parquet_path, combined_data)
            
            # Update episodes.jsonl
            self._update_episodes_metadata(combined_data)
            
            # Print summary
            print(f"Episode {self.episode_index} saved to {self.dataset_root}")
            print(f"  Frames: {len(combined_data)}")
            print(f"  Video: {video_path}")
            print(f"  Data: {parquet_path}")
            print(f"  Total reward: {sum(step['next.reward'] for step in combined_data):.4f}")
            print(f"  Success: {combined_data[-1]['next.success']}")
            
            # Increment episode index
            self.episode_index += 1
            
        except Exception as e:
            print(f"Error saving episode: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_video(self, video_path: Path, frames: List[np.ndarray]) -> None:
        """Encode frames as MP4 video."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        
        writer.release()
    
    def _save_parquet(self, parquet_path: Path, data: List[Dict[str, Any]]) -> None:
        """Save numerical data as Parquet file."""
        df = pd.DataFrame(data)
        
        # Convert lists to arrays for proper parquet storage
        for col in df.columns:
            if df[col].dtype == object:
                first_val = df[col].iloc[0]
                if isinstance(first_val, list):
                    df[col] = df[col].apply(
                        lambda x: np.array(x, dtype=np.float32) if x is not None else None
                    )
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)
    
    def _update_episodes_metadata(self, data: List[Dict[str, Any]]) -> None:
        """Append episode metadata to episodes.jsonl."""
        episodes_path = self.meta_dir / "episodes.jsonl"
        
        episode_meta = {
            "episode_index": self.episode_index,
            "tasks": [data[0]['task_index']],
            "length": len(data),
            "timestamp": time.time(),
            "success": data[-1]['next.success'],
            "total_reward": sum(step['next.reward'] for step in data)
        }
        
        with open(episodes_path, 'a') as f:
            f.write(json.dumps(episode_meta) + '\n')
    
    def reset(self) -> None:
        """Reset both wrappers for new episode."""
        self.image_wrapper.reset()
        self.robot_wrapper.reset()
    
    def discard(self) -> None:
        """Discard current episode data."""
        self.image_wrapper.discard()
        self.robot_wrapper.discard()
    
    def connect(self) -> None:
        """Connect both devices."""
        self.image_wrapper.connect()
        self.robot_wrapper.connect()
    
    def close(self) -> None:
        """Close both devices."""
        self.image_wrapper.close()
        self.robot_wrapper.close()


class ImageDataWrapper:
    """Collects and saves only image data as videos."""
    
    def __init__(
        self, 
        camera_device: CameraDevice, 
        fps: int = 30, 
        video_codec: str = 'mp4v',
        output_dir: str = "collected_data"
    ) -> None:
        self.camera_device = camera_device
        self.fps = fps
        self.video_codec = video_codec
        self.episode_frames: List[np.ndarray] = []
        self.episode_start_time = None
        self.episode_index = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_step(self) -> None:
        """Capture and store a single image frame."""
        if self.episode_start_time is None:
            self.episode_start_time = time.time()
        
        frame = self.camera_device.get_image()
        if frame is not None:
            self.episode_frames.append(frame)
            print(f"[DEBUG] {self.camera_device._name}: Frame captured, shape={frame.shape}")
        else:
            print(f"[DEBUG] {self.camera_device._name}: No frame received (get_image returned None)")
    
    def save_episode(self) -> None:
        """Save collected frames as a video file."""
        if not self.episode_frames:
            print(f"{self.camera_device._name}: No frames to save")
            return
        
        video_path = self.output_dir / f"{self.camera_device._name}_episode_{self.episode_index:06d}.mp4"
        self._save_video(video_path)
        
        print(f"{self.camera_device._name}: Episode {self.episode_index} saved with {len(self.episode_frames)} frames to {video_path}")
        self.episode_index += 1
    
    def _save_video(self, video_path: Path) -> None:
        """Encode frames as MP4 video."""
        if not self.episode_frames:
            return
        
        height, width = self.episode_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        
        for frame in self.episode_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        
        writer.release()
    
    def get_frames(self) -> List[np.ndarray]:
        """Return collected frames."""
        return self.episode_frames
    
    def reset(self) -> None:
        """Clear frames for new episode."""
        self.episode_frames = []
        self.episode_start_time = None
    
    def discard(self) -> None:
        """Discard frames without saving."""
        self.episode_frames = []
        self.episode_start_time = None
    
    def connect(self) -> None:
        """Connect to camera device."""
        pass
    
    def close(self) -> None:
        """Close camera device."""
        pass


class RobotDataWrapper:
    """Collects and saves only robot state/proprioception data."""
    
    def __init__(self, arm: RemoteFranka, gripper: RemoteFrankaGripper) -> None:
        self.arm = arm
        self.gripper = gripper
        self.episode_data: List[Dict[str, Any]] = []
        self.episode_start_time = None

    def _get_proprioception(self, observation_state: Optional[List[float]]) -> List[float]:
        """Get robot proprioception state."""
        if observation_state is not None:
            return list(map(float, observation_state))

        try:
            robot_state = self.arm.get_state()
            return list(map(float, robot_state.get("pose", [0.0, 0.0])[:2]))
        except Exception:
            return [0.0, 0.0]

    def _normalize_action(self, action: List[float]) -> List[float]:
        """Normalize action to list of floats."""
        if isinstance(action, list):
            return list(map(float, action))
        if hasattr(action, "__iter__"):
            return list(map(float, action))
        return [float(action)]

    def save_step(
        self,
        action: List[float],
        observation_state: Optional[List[float]] = None,
        reward: float = 0.0,
        done: bool = False,
        success: bool = False,
        task_index: int = 0,
    ) -> None:
        """Save robot state for a single step."""
        if self.episode_start_time is None:
            self.episode_start_time = time.time()
        current_time = time.time() - self.episode_start_time

        observation_state = self._get_proprioception(observation_state)
        action = self._normalize_action(action)

        step_data = {
            "observation.state": observation_state,
            "action": action,
            "frame_index": len(self.episode_data),
            "timestamp": float(current_time),
            "next.reward": float(reward),
            "next.done": bool(done),
            "next.success": bool(success),
            "task_index": int(task_index),
        }

        self.episode_data.append(step_data)
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Return collected data."""
        return self.episode_data

    def reset(self) -> None:
        """Clear episode data."""
        self.episode_data = []
        self.episode_start_time = None

    def discard(self) -> None:
        """Discard data without saving."""
        self.episode_data = []
        self.episode_start_time = None
    
    def connect(self) -> None:
        """Connect to robot."""
        pass
    
    def close(self) -> None:
        """Close robot connection."""
        pass
