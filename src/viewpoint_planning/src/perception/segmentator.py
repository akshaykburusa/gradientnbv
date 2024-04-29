import torch
import cv2
import rospy
import ros_numpy
import numpy as np
import time

from sensor_msgs.msg import Image

# Detectron2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances

from memory_profiler import profile


class InstanceSegmentator:
    """
    Perform instance segmentation using Mask R-CNN.
    """

    def __init__(self, threshold=None):
        """
        Initialize the segmentator.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Detectron2 and Mask R-CNN parameters.
        setup_logger()
        self.cfg = get_cfg()
        self.cfg.merge_from_file(rospy.get_param("/mask_rcnn/model"))
        if threshold is not None:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        else:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = rospy.get_param(
                "/mask_rcnn/detection_threshold"
            )
        self.cfg.MODEL.WEIGHTS = rospy.get_param("/mask_rcnn/weights")
        self.cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = rospy.get_param("/mask_rcnn/classes")
        MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes = self.class_names
        # ROS topics.
        self.vis_pub = rospy.Publisher(
            "segmentation/visualization", Image, queue_size=1
        )
        self.reset()

    def reset(self):
        self.counter = 0

    def get_detections(self, color_image: np.ndarray):
        if color_image is None:
            rospy.logwarn("[Segmentator] Segmentation failed. No data from camera.")
            return
        # Perform instance segmentation with Mask R-CNN.
        outputs = self.predictor(color_image)
        predictions = outputs["instances"].to("cpu")
        class_ids, boxes, masks, scores = self.extract_instances(predictions)
        return predictions, class_ids, boxes, masks, scores

    def get_semantics(
        self,
        class_ids: torch.tensor,
        masks: torch.tensor,
        scores: torch.tensor,
        image_size: tuple,
        classes_of_interest: list,
        depth_image: np.ndarray,
    ) -> torch.tensor:
        """
        Get masks of class labels and confidence scores of all instances from the output of instance segmentation in Detectron2
        :param predictions: output of instance segmentation in Detectron2
        :return: tensor of shape (H, W, 2) with class labels and confidence scores
        """
        if class_ids is None or masks is None or image_size is None:
            return
        class_ids = class_ids.to(torch.float32).to(self.device)
        scores = scores.to(self.device)
        # TODO: Fix label mask to default class
        label_mask = -1 * torch.ones(
            image_size, dtype=torch.float32, device=self.device
        )
        # Create a mask that is log odds 0.9 if there's a semantic value and log odds of 0.01 otherwise
        free_value = self.log_odds(0.1)
        occupied_value = self.log_odds(0.7)
        # # For octomap-based active vision, use probabilities instead of log odds
        # free_value = 0.4
        score_mask = free_value * torch.ones(
            image_size, dtype=torch.float32, device=self.device
        )
        for i in range(masks.shape[0]):
            if class_ids[i] not in classes_of_interest:
                continue
            # get indices of depth values that are not too far from the median
            depth_map = -1.0 * np.ones(image_size, dtype=np.float32)
            depth_map[masks[i, :, :]] = depth_image[masks[i, :, :]]
            depth_median = np.nanmedian(depth_map[masks[i, :, :]])
            depth_indices = np.where(np.abs(depth_map - depth_median) <= 0.03)
            # # assign semantic values to the depth indices
            # occupied_value = scores[i]
            score_mask[depth_indices] = occupied_value
            label_mask[depth_indices] = class_ids[i]
        semantics = torch.stack((score_mask, label_mask), dim=-1)
        return semantics

    def extract_instances(self, predictions: Instances):
        """
        Extract instances from prediction output of Mask R-CNN
        :param predictions: output of instance segmentation in Detectron2
        :return: class_ids, boxes, masks, scores
        """
        # Return if no data.
        if predictions is None:
            rospy.logwarn("[Segmentator] Prediction extraction failed.")
            return
        # Extract predictions.
        class_ids = (
            predictions.pred_classes if predictions.has("pred_classes") else None
        )
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        masks = predictions.pred_masks if predictions.has("pred_masks") else None
        scores = predictions.scores if predictions.has("scores") else None
        return class_ids, boxes, masks, scores

    def log_odds(self, p):
        return np.log(p / (1 - p))

    def visualize(
        self,
        color_image: np.ndarray,
        predictions: Instances,
        save_path=None,
        timestamp=None,
    ):
        """
        Visualize the detected instances.
        :param color_image: color image
        :param predictions: output of instance segmentation in Detectron2
        """
        v = Visualizer(
            color_image[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.2,
        )

        # Visualize only
        out = v.draw_instance_predictions(predictions[predictions.pred_classes != 2])
        # out = v.draw_instance_predictions(predictions)
        img = out.get_image()[:, :, ::-1]
        if save_path is not None:
            self.counter += 1
            # cv2.imwrite(
            #     save_path + "col_" + str(self.counter) + ".png",
            #     color_image,
            # )
            # cv2.imwrite(
            #     save_path + "sem_" + str(self.counter) + ".png",
            #     img,
            # )
            cv2.imwrite(
                save_path + timestamp + "_seg.png",
                img,
            )

        # save the color and depth images using cv2
        save_path = "/home/akshay/Code/paper3_ws/manuscript"
        cv2.imwrite(save_path + "/seg_image.png", img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_msg = ros_numpy.msgify(Image, img, encoding="rgb8")
        self.vis_pub.publish(image_msg)
