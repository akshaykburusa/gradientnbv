import rospy
import numpy as np
import ros_numpy

from sensor_msgs.msg import CameraInfo, Image, PointCloud2


class RealsenseROSCapturer:
    """
    Gets the color and depth frames of a realsense RGB-D camera from ROS (D4XX, D5XX).
    """

    def __init__(self):
        # Color and depth frames.
        self.color_image = None
        self.depth_image = None
        self.points = None
        self.camera_info = None
        # Realsense ROS topics.
        self.use_sim = rospy.get_param("/use_sim_time", True)
        rospy.Subscriber(
            "/camera/color/image_rect_color", Image, self.color_callback, queue_size=2
        )
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info",
            CameraInfo,
            self.info_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/camera/depth_registered/points",
            PointCloud2,
            self.points_callback,
            queue_size=1,
            buff_size=100000000,
        )

    def color_callback(self, msg):
        # Subscriber callback for color image.
        data = ros_numpy.numpify(msg)
        self.color_image = data[:, :, ::-1]

    def depth_callback(self, msg):
        # Subscriber callback for depth image.
        data = ros_numpy.numpify(msg)
        if not self.use_sim:
            data = data.astype("float32") / 1000.0
        self.depth_image = data

    def points_callback(self, msg):
        # Subscriber callback for point cloud.
        data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)
        self.points = np.reshape(data, (msg.height, msg.width, 3))
        if self.use_sim:
            self.points[..., 1] += 0.024

    def info_callback(self, msg):
        # Subscriber callback for camera info.
        self.camera_info = msg

    def get_frames(self):
        # Get the next color and depth frame
        color_output = {}
        depth_output = {}
        color_output["color_image"] = self.color_image
        depth_output["depth_image"] = self.depth_image
        depth_output["points"] = self.points
        return self.camera_info, color_output, depth_output
