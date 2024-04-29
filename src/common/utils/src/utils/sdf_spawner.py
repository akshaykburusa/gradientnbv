import rospy
import rospkg
import numpy as np

from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler


class SDFSpawner:
    def __init__(self, model_name="box"):
        # ROS service for spawning.
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.model_name = model_name
        rospack = rospkg.RosPack()
        self.model_path = rospack.get_path("simulation_environment") + "/sdfs/"

    def spawn_box(self, pos):
        pos = pos - np.array([0.0, 0.0, 0.024])  # Correcting for Gazebo coordinates
        box_pose = Pose(
            position=Point(*pos),
            orientation=Quaternion(*quaternion_from_euler(0, 0, 0)),
        )
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            spawner(
                model_name="box",
                model_xml=open(
                    self.model_path + "box.sdf",
                    "r",
                ).read(),
                robot_namespace="",
                initial_pose=box_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def delete_box(self):
        rospy.wait_for_service("gazebo/delete_model")
        delete_model_service = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        try:
            delete_model_service(model_name="box")
        except Exception as e:
            print("delete box failed")
