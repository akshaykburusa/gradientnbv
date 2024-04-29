#!/usr/bin/env python3

import rospy
import random

from geometry_msgs.msg import Pose, Point, Quaternion
from abb_control.srv import ArmGoal


def RandomPoseGenerator(minx, maxx, miny, maxy, minz, maxz):
    return Pose(
        Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
            random.uniform(minz, maxz),
        ),
        Quaternion(0.0, 0.0, 0.0, 1.0),
    )


class ArmControlClient:
    def __init__(self):
        # Initialize the subscribers/publishers
        rospy.loginfo("[ArmControlClient] Initializing arm control test")

        # Initialize the services
        self.armcontrol_service = rospy.ServiceProxy("move_arm_to_pose", ArmGoal)

        # define workspace borders
        self.minx = 0.40
        self.maxx = 0.80
        self.miny = -0.40
        self.maxy = 0.40
        self.minz = 1.00
        self.maxz = 1.40

        # Check / wait for communication to motor drives being operational
        rospy.loginfo("[ArmControlClient] Waiting for the sercives to become available")
        rospy.wait_for_service("move_arm_to_pose")
        rospy.loginfo("[ArmControlClient] Arm control node has started")

        test_pose = RandomPoseGenerator(
            self.minx, self.maxx, self.miny, self.maxy, self.minz, self.maxz
        )

    def move_arm_to_pose(self, pose):
        try:
            result = self.armcontrol_service(pose)
            # if result:
            #     rospy.loginfo("[ArmControlClient] Arm arrived at pose")
            # else:
            #     rospy.logerr("[ArmControlClient] Arm motion failed")
            return result
        except rospy.ServiceException as e:
            print("[ArmControlClient] Service call failed: %s" % e)
            return False
