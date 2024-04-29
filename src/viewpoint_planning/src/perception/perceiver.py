import rospy
import cv2
import torch
import numpy as np

from perception.realsense_ros_capturer import RealsenseROSCapturer


class Perceiver:
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """

    def __init__(self):
        self.capturer = RealsenseROSCapturer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_camera_info(self):
        camera_info, _, _ = self.capturer.get_frames()
        return camera_info

    def run(self):
        # Get data from camera
        camera_info, color_output, depth_output = self.capturer.get_frames()
        color_image = color_output["color_image"]
        depth_image = depth_output["depth_image"]
        points = depth_output["points"]
        # Return if no data
        if camera_info is None or color_image is None:
            rospy.logwarn("[Perceiver] Perception paused. No data from camera.")
            return
        # Color-based segmentation
        # Note: Only for the toy example. Replace with an object-detection network in practice.
        segmentation_mask = self.color_segmentation(color_image)
        # Get semantics
        semantics = self.assign_semantics(camera_info, segmentation_mask)
        return depth_image, points, semantics

    def color_segmentation(self, color_image: np.array) -> np.array:
        """
        Perform color segmentation on the input image using OpenCV
        :param color_image: input color image
        :return: segmentation mask
        """
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # Define range of red color in HSV
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([10, 255, 255])
        # Threshold the HSV image to get only white colors
        segmentation_mask = cv2.inRange(hsv_image, lower_color, upper_color)
        return segmentation_mask

    def assign_semantics(self, camera_info, segmentation_mask) -> torch.tensor:
        """
        Assign the confidence scores and labels to the pixels in the image
        :param camera_info: camera information
        :param segmentation_mask: segmentation mask [H x W]
        :return: semantic confidence scores and labels [H x W x 2]
        """
        # Image size
        image_size = (camera_info.height, camera_info.width)
        # Create a mask that is log odds 0.9 if there's a semantic value and log odds of 0.4 otherwise
        occupied_odds = self.log_odds(0.9)
        free_odds = self.log_odds(0.4)
        # Initialize the label mask as free
        score_mask = free_odds * torch.ones(
            image_size, dtype=torch.float32, device=self.device
        )
        # Initialize the label mask as background
        label_mask = -1 * torch.ones(
            image_size, dtype=torch.float32, device=self.device
        )
        # Assign the semantic labels
        score_mask[segmentation_mask > 0] = occupied_odds
        label_mask[segmentation_mask > 0] = 0
        semantics = torch.stack((score_mask, label_mask), dim=-1)
        return semantics

    def log_odds(self, p):
        return np.log(p / (1 - p))
