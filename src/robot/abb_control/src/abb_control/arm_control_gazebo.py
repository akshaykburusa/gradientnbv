#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import tf2_ros
from geometry_msgs.msg import TransformStamped


class ArmControlGazebo:
    def __init__(self):
        """
        Constructor
        """
        # ROS service to get and set model state.
        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")
        # Parameters.
        self.br = tf2_ros.TransformBroadcaster()
        self.arm_name = "abb_l515"
        self.arm_frame = "base_link"
        self.ref_frame = "world"

    def get_agent_pose(self):
        get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        model_state = get_state(self.arm_name, self.ref_frame)
        if model_state:
            return model_state
        else:
            print("[ArmControlGazebo] Failed to get model state")

    def move_agent_to_pose(self, pose):
        state_msg = ModelState()
        state_msg.model_name = self.arm_name
        state_msg.pose = pose
        state_msg.reference_frame = self.ref_frame
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state_msg)
        except rospy.ServiceException as e:
            print("[ArmControlGazebo] Service call to set_model_state failed: ", e)

    def broadcast(self, pose):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.ref_frame
        t.child_frame_id = self.arm_frame
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        self.br.sendTransform(t)
