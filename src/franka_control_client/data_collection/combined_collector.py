"""Combined data collector that merges image and robot data in LeRobot format."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cv2


class CombinedDataCollector:
    """Combines image and robot data, saves in LeRobot dataset format."""
    
    def __init__(
        self,
        image_wrapper,  # ImageDataWrapper
        robot_wrapper,  # RobotDataWrapper
        dataset_root: str = "LeRobotDataset",
        fps: int = 30,
        video_codec: str = 'mp4v',
        chunk_size: int = 1000
    ):
        self.image_wrapper = image_wrapper
        self.robot_wrapper = robot_wrapper
        self.fps = fps
        self.video_codec = video_codec
        self.chunk_size = chunk_size
        
        # Episode tracking
        self.episode_index = 0
        self.global_index = 0
        
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
        self.robot_wrapper.save_step(
            action=action,
            observation_state=observation_state,
            reward=reward,
            done=done,
            success=success,
            task_index=task_index
        )
    
    def save_episode(self) -> None:
        """Combine and save episode data in LeRobot format."""
        frames = self.image_wrapper.get_frames()
        robot_data = self.robot_wrapper.get_data()
        
        if not frames or not robot_data:
            print("No data to save")
            return
        
        if len(frames) != len(robot_data):
            print(f"Warning: Frame count ({len(frames)}) != robot data count ({len(robot_data)})")
            # Truncate to minimum length
            min_len = min(len(frames), len(robot_data))
            frames = frames[:min_len]
            robot_data = robot_data[:min_len]
        
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
            
            # Merge robot data with episode/index info and save as parquet
            combined_data = self._merge_data(robot_data)
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
            
            # Increment episode counter
            self.episode_index += 1
            
        except Exception as e:
            print(f"Error saving episode: {e}")
            import traceback
            traceback.print_exc()
    
    def _merge_data(self, robot_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add episode_index and index to robot data."""
        merged = []
        for step in robot_data:
            step['episode_index'] = self.episode_index
            step['index'] = self.global_index
            merged.append(step)
            self.global_index += 1
        return merged
    
    def _save_video(self, video_path: Path, frames: List[np.ndarray]) -> None:
        """Encode frames as MP4 video."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
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
