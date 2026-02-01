"""
Example usage of LeRobotDataCollector for LeRobot dataset collection.

This example shows how to use LeRobotDataCollector which combines
image and robot data internally in its save_episode method.
"""

from franka_control_client.camera.camera import CameraDevice
from franka_control_client.franka_robot.franka_arm import RemoteFranka
from franka_control_client.franka_robot.franka_gripper import RemoteFrankaGripper
from franka_control_client.data_collection import LeRobotDataCollector


def main():
    # Initialize devices
    camera = CameraDevice(camera_ip="192.168.1.100", port=5000)
    robot_arm = RemoteFranka(robot_ip="192.168.1.101")
    robot_gripper = RemoteFrankaGripper(robot_ip="192.168.1.101")
    
    # Create LeRobot data collector (combines image + robot data)
    collector = LeRobotDataCollector(
        camera_device=camera,
        robot_arm=robot_arm,
        robot_gripper=robot_gripper,
        dataset_root="LeRobotDataset",
        fps=30,
        chunk_size=1000
    )
    
    # Connect to devices
    collector.connect()
    
    try:
        # Collect multiple episodes
        for episode_num in range(5):
            print(f"\n=== Episode {episode_num} ===")
            print("Recording...")
            
            # Simulate episode data collection
            for step in range(100):  # 100 steps per episode
                # Get action from your policy/teleoperation
                action = [0.1 * step, 0.2 * step]  # Example action
                
                # Get robot state (optional, will auto-fetch if None)
                observation_state = None  # Or provide: [x, y] from robot
                
                # Calculate reward (from your task)
                reward = 0.01 * step
                
                # Check if done
                done = (step == 99)
                success = (step == 99 and reward > 0.5)
                
                # Save step (captures image + robot state)
                collector.save_step(
                    action=action,
                    observation_state=observation_state,
                    reward=reward,
                    done=done,
                    success=success,
                    task_index=0
                )
            
            # Save episode (combines image and robot data in LeRobot format)
            collector.save_episode()
            
            # Reset for next episode
            collector.reset()
            
            print(f"Episode {episode_num} completed!")
    
    finally:
        # Clean up
        collector.close()
        print("\nDataset saved to LeRobotDataset/")
        print("Structure:")
        print("  LeRobotDataset/")
        print("  ├── data/chunk-000/episode_*.parquet")
        print("  ├── videos/chunk-000/observation.image/episode_*.mp4")
        print("  └── meta/")
        print("      ├── episodes.jsonl")
        print("      └── info.json")


if __name__ == "__main__":
    main()
