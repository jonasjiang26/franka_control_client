# LeRobot Dataset Collection Implementation

## Summary

Modified the data collection system to save robot demonstrations in LeRobot dataset format with separate image (video) and numerical data storage.

## Architecture

### 1. **ImageDataWrapper** (Image-only collection)
- **Purpose**: Captures and stores only camera images
- **Methods**:
  - `save_step()`: Captures a single frame
  - `get_frames()`: Returns all collected frames
  - `reset()`: Clears frames for new episode
  
### 2. **RobotDataWrapper** (State-only collection)
- **Purpose**: Captures and stores only robot state/proprioception
- **Methods**:
  - `save_step()`: Records robot state, action, rewards, etc.
  - `get_data()`: Returns all collected state data
  - `reset()`: Clears data for new episode

### 3. **CombinedDataCollector** (Unified collection)
- **Purpose**: Combines image + robot data, saves in LeRobot format
- **Methods**:
  - `save_step()`: Saves to both wrappers simultaneously
  - `save_episode()`: Merges data and saves in LeRobot structure
  - `reset()`: Resets both wrappers

## Dataset Structure

```
LeRobotDataset/
├── data/                           # Parquet files with numerical data
│   ├── chunk-000/
│   │   ├── episode_000000.parquet  # States, actions, timestamps
│   │   ├── episode_000001.parquet
│   │   └── episode_NNNNNN.parquet
│   ├── chunk-001/
│   │   └── episode_*.parquet
│   └── ...
├── videos/                         # MP4 files for camera feeds
│   ├── chunk-000/
│   │   └── observation.image/
│   │       ├── episode_000000.mp4
│   │       ├── episode_000001.mp4
│   │       └── ...
│   ├── chunk-001/
│   └── ...
└── meta/                           # Metadata and configuration
    ├── episodes.jsonl              # Episode boundaries and metadata
    └── info.json                   # Dataset schema, shapes, fps
```

## Usage Example

```python
from franka_control_client.data_collection import (
    ImageDataWrapper,
    RobotDataWrapper,
    CombinedDataCollector
)

# Initialize wrappers
image_wrapper = ImageDataWrapper(camera_device=camera, fps=30)
robot_wrapper = RobotDataWrapper(arm=robot_arm, gripper=gripper)

# Combine
collector = CombinedDataCollector(
    image_wrapper=image_wrapper,
    robot_wrapper=robot_wrapper,
    dataset_root="LeRobotDataset"
)

# Collect episode
for step in range(100):
    collector.save_step(
        action=[x, y],
        observation_state=[ee_x, ee_y],
        reward=0.5,
        done=False,
        success=False
    )

# Save episode
collector.save_episode()
collector.reset()
```

## Data Format

### Parquet Columns
- `observation.state`: float32 array (proprioception)
- `action`: float32 array
- `timestamp`: float32
- `next.reward`: float32
- `next.done`: bool
- `next.success`: bool
- `episode_index`: int64
- `frame_index`: int64
- `index`: int64 (global step counter)
- `task_index`: int64

### Video Format
- Codec: mp4v (configurable)
- FPS: 30 (configurable)
- Synchronized with parquet data by frame_index

## Files Modified/Created

1. **Modified**: `src/franka_control_client/data_collection/utils.py`
   - Simplified ImageDataWrapper (image-only)
   - Simplified RobotDataWrapper (state-only)

2. **Created**: `src/franka_control_client/data_collection/combined_collector.py`
   - CombinedDataCollector class

3. **Created**: `src/franka_control_client/data_collection/__init__.py`
   - Module exports

4. **Created**: `examples/collect_lerobot_dataset.py`
   - Usage example

## Key Features

✅ Separate image and state collection
✅ Combined saving in LeRobot format
✅ Chunked organization (1000 episodes per chunk)
✅ Video compression (MP4)
✅ Parquet for numerical data
✅ Metadata tracking (episodes.jsonl, info.json)
✅ Synchronized frame indexing
✅ Episode-level statistics
