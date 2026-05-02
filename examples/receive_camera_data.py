import pyzlc
from franka_control_client.camera.camera import CameraDevice
from franka_control_client.data_collection.irl_wrapper import (
    ImageDataWrapper,
)
from lang_sam import LangSAM
import numpy as np
import cv2
from PIL import Image

if __name__ == "__main__":
    pyzlc.init(
        "policy_inference",
        "141.3.53.25",
        group_name="robot_lab_robotiq_202",
        group_port=7725,
    )
    # static_cam = ImageDataWrapper(CameraDevice("static_cam", preview=False), hw_name="static_cam")
    wrist_cam = ImageDataWrapper(CameraDevice("wrist_cam", preview=False), hw_name="wrist_cam")
    sam = LangSAM()
    try:
        while True:
            frame = wrist_cam.capture_step()
            rgb_image, depth_image = frame


            text_prompt = "bowl"
            image_pil = Image.fromarray(rgb_image).convert("RGB")
            results = sam.predict([image_pil], [text_prompt])
            segmentation_mask = results[0]["masks"]
            if segmentation_mask.ndim == 3:
                segmentation_mask = np.any(segmentation_mask, axis=0)

            # Visualize the segmentation mask
            color_mask = np.zeros_like(rgb_image)
            color_mask[segmentation_mask == 1] = [0, 255, 0]  # Green color for the segmented object
            blended_image = cv2.addWeighted(rgb_image, 0.7, color_mask, 0.3, 0)
            blended_image = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            # cv2.imwrite("segmentation_mask.png", blended_image)  # Save the blended image with segmentation mask
            cv2.imshow("Segmentation Mask", blended_image)  # Show mask in binary format
            cv2.waitKey(1)

        

    except KeyboardInterrupt:
        pass
