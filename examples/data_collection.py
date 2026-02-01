import pyzlc
from franka_control_client.data_collection.data_collection_manager import (
    DataCollectionManager,
)
from franka_control_client.data_collection.utils import ImageDataWrapper
from franka_control_client.camera.camera import CameraDevice

if __name__ == "__main__":
    # Initialize pyzlc communication node
    pyzlc.init(
        "data_collection_client",
        "10.172.218.210",
        group_name="hardware_collection",
    )
    
    centric_cam = CameraDevice(camera_name="/centric_camera", preview=True)
    wrist_cam = CameraDevice(camera_name="/wrist_camera", preview=True)
    
    centric_collector = ImageDataWrapper(centric_cam)
    wrist_collector = ImageDataWrapper(wrist_cam)

    data_collection_manager = DataCollectionManager(
        [centric_collector, wrist_collector]
    )
    data_collection_manager.run()
