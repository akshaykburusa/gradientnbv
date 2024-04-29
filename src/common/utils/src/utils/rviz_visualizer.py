# ROS node to visualize topics in rviz

import rospy
import numpy as np
import ros_numpy
import struct

from copy import deepcopy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point, Pose, PoseArray
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge


class RvizVisualizer:
    def __init__(self):
        # Frames
        self.world_frame_id = rospy.get_param("/agent/frames/world_frame_id")
        self.camera_frame_id = rospy.get_param("/agent/frames/camera_frame_id")
        # Initialize the rviz visualizer
        self.view_samples_pub = rospy.Publisher(
            "view_samples", MarkerArray, queue_size=1
        )
        self.viewpoint_pub = rospy.Publisher("viewpoint", Marker, queue_size=1)
        self.pc2_pub = rospy.Publisher("voxels", PointCloud2, queue_size=1)
        self.rois_pub = rospy.Publisher("rois", MarkerArray, queue_size=1)
        self.camera_bounds_pub = rospy.Publisher("camera_bounds", Marker, queue_size=1)
        self.semantic_mean_pub = rospy.Publisher("semantic_mean", Marker, queue_size=1)
        self.world_model_pub = rospy.Publisher(
            "world_model/objects", MarkerArray, queue_size=1
        )
        self.point_cloud_pub = rospy.Publisher(
            "gt_point_cloud", PointCloud2, queue_size=1
        )
        self.poses_with_covariance_pub = rospy.Publisher(
            "poses_with_covariance", MarkerArray, queue_size=1
        )
        self.poses_pub = rospy.Publisher(
            "pose_estimation/poses", PoseArray, queue_size=1
        )
        self.point_pub = rospy.Publisher("point", Marker, queue_size=1)
        self.class_ids_pub = rospy.Publisher("class_ids", MarkerArray, queue_size=1)
        self.curve_pub = rospy.Publisher("curve", Marker, queue_size=1)
        self.points_pub = rospy.Publisher("points", Marker, queue_size=1)
        self.pred_points_pub = rospy.Publisher("pred_points", Marker, queue_size=1)
        self.point_cloud_pub2 = rospy.Publisher(
            "true_point_cloud", PointCloud2, queue_size=1
        )
        self.gain_image_pub = rospy.Publisher("gain_image", Image, queue_size=1)

    def visualize_view_samples(self, view_samples: PoseArray) -> None:
        """
        Visualize the view samples in Rviz as a MarkerArray message
        :param view_samples: PoseArray of view samples
        """
        marker_array = MarkerArray()
        for i, pose in enumerate(view_samples.poses):
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "view_samples"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(0.02, 0.01, 0.01)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
            marker_array.markers.append(marker)
        self.view_samples_pub.publish(marker_array)

    def visualize_viewpoint(self, viewpoint: Pose) -> None:
        """
        Visualize the viewpoint in Rviz as a Marker message
        :param viewpoint: Pose of the viewpoint
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "viewpoint"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = viewpoint
        marker.scale = Vector3(0.02, 0.01, 0.01)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        self.viewpoint_pub.publish(marker)

    def visualize_voxels(
        self, points: np.array, semantics: np.array, class_ids: np.array
    ) -> None:
        """
        Visualize the voxels in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param semantics: (N, 1) array of semantics
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        color_occ = struct.unpack("I", struct.pack("BBBB", 60, 111, 2, 255))[0]
        # color_sem = struct.unpack("I", struct.pack("BBBB", 0, 255, 0, 255))[0]
        color_0 = struct.unpack("I", struct.pack("BBBB", 0, 0, 255, 255))[0]
        color_1 = struct.unpack("I", struct.pack("BBBB", 0, 0, 255, 255))[0]
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        )
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = color_occ
        for i in range(points.shape[0]):
            if class_ids[i] == 0:
                points_arr["rgb"][i] = color_0
            elif class_ids[i] == 1:
                points_arr["rgb"][i] = color_1

        # Convert the NumPy array to a PointCloud2 message
        voxel_points = ros_numpy.point_cloud2.array_to_pointcloud2(
            points_arr, rospy.Time.now(), self.world_frame_id
        )
        self.pc2_pub.publish(voxel_points)

    def visualize_rois(self, rois: PoseArray) -> None:
        """
        Visualize the ROIs in Rviz as a MarkerArray message
        :param rois: PoseArray of ROIs
        """
        marker_array = MarkerArray()
        for i, pose in enumerate(rois.poses):
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "rois"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(0.09, 0.09, 0.09)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
            marker_array.markers.append(marker)
        self.rois_pub.publish(marker_array)

    def visualize_camera_bounds(self, bounds: np.array) -> None:
        """
        Visualize the camera bounds in Rviz as a Marker message
        :param bounds: (2, 3) array of camera bounds
        """
        center = np.mean(bounds, axis=0)
        size = np.max(bounds, axis=0) - np.min(bounds, axis=0)
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_bounds"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(*center)
        marker.scale = Vector3(*size)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.3)
        self.camera_bounds_pub.publish(marker)

    def visualize_semantic_mean(self, mean: np.array) -> None:
        """
        Visualize the semantic mean in Rviz as a Marker message
        :param mean: (3, ) array of semantic mean
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "semantic_mean"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(*mean)
        marker.scale = Vector3(0.02, 0.02, 0.02)
        marker.color = ColorRGBA(1.0, 0.0, 1.0, 0.8)
        self.semantic_mean_pub.publish(marker)

    def visualize_wm(self, markers: MarkerArray) -> None:
        """
        Visualize the world model in Rviz as a MarkerArray message
        :param markers: MarkerArray of the objects in the world model
        """
        self.world_model_pub.publish(markers)

    def visualize_point_cloud(self, points: np.array, color: np.array) -> None:
        """
        Visualize the point cloud in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param color: (N, 3) array of colors
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        assert color.shape[1] == 3, "The input array must have shape (N, 3)"
        assert (
            points.shape[0] == color.shape[0]
        ), "The input arrays must have the same length"
        color = (color * 255).astype(np.uint8)
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        )
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = np.array(
            [
                struct.unpack("I", struct.pack("BBBB", *color[i, ::-1], 255))[0]
                for i in range(color.shape[0])
            ]
        )
        # Convert the NumPy array to a PointCloud2 message
        point_cloud = ros_numpy.point_cloud2.array_to_pointcloud2(
            points_arr, rospy.Time.now(), "world"
        )
        self.point_cloud_pub.publish(point_cloud)

    def visualize_gt_point_cloud(self, points: np.array, color: np.array) -> None:
        """
        Visualize the point cloud in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param color: (N, 3) array of colors
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        assert color.shape[1] == 3, "The input array must have shape (N, 3)"
        assert (
            points.shape[0] == color.shape[0]
        ), "The input arrays must have the same length"
        color = (color * 255).astype(np.uint8)
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        )
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = np.array(
            [
                struct.unpack("I", struct.pack("BBBB", *color[i, ::-1], 255))[0]
                for i in range(color.shape[0])
            ]
        )
        # Convert the NumPy array to a PointCloud2 message
        point_cloud = ros_numpy.point_cloud2.array_to_pointcloud2(
            points_arr, rospy.Time.now(), "world"
        )
        self.point_cloud_pub2.publish(point_cloud)

    def visualize_curve(self, x, y, z):
        """
        Visualize the curve in Rviz as a Marker message
        :param x: (N, ) array of x coordinates
        :param y: (N, ) array of y coordinates
        :param z: (N, ) array of z coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "curve"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.02, 0.02, 0.02)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
        for i in range(len(x)):
            marker.points.append(Point(x[i], y[i], z[i]))
        self.curve_pub.publish(marker)

    def visualize_point(self, point: np.array):
        """
        Visualize the point in Rviz as a Marker message
        :param point: (3, ) array of point coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(*point)
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.02, 0.02, 0.02)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        self.point_pub.publish(marker)

    def visualize_points(self, points: np.ndarray):
        """
        Visualize the points in Rviz as a Marker message
        :param points: (N, 3) array of point coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.01, 0.01, 0.01)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        for point in points:
            marker.points.append(Point(*point))
        self.points_pub.publish(marker)

    def visualize_pred_points(self, points: np.ndarray):
        """
        Visualize the points in Rviz as a Marker message
        :param points: (N, 3) array of point coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pred_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.02, 0.02, 0.02)
        marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)
        for point in points:
            marker.points.append(Point(*point))
        self.pred_points_pub.publish(marker)

    def visualize_gain_image(self, image: np.ndarray):
        """
        Visualize the gain image in Rviz as Image message
        :param image: (H, W, 3) array of gain image
        """
        bridge = CvBridge()
        image = (image * 255).astype(np.uint8)[:, :, ::-1]
        image_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
        image_msg.header.frame_id = self.camera_frame_id
        image_msg.header.stamp = rospy.Time.now()
        self.gain_image_pub.publish(image_msg)
