#!/usr/bin/env python3

import rospy
import random
from geometry_msgs.msg import Pose, Point, Quaternion

from abb_control.arm_control_client import ArmControlClient


def RandomPoseGenerator(minx, maxx, miny, maxy, minz, maxz):
    return Pose(
        Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
            random.uniform(minz, maxz),
        ),
        Quaternion(0.0, 0.0, 0.0, 1.0),
    )


if __name__ == "__main__":
    # Initiate task_manager node.
    rospy.init_node("arm_control_test")

    # Object instantiations.
    arm_control_client = ArmControlClient()

    # Declarations.
    minx = 0.3
    maxx = 0.6
    miny = -0.3
    maxy = 0.3
    minz = 0.8
    maxz = 1.2

    while not rospy.is_shutdown():
        test_pose = RandomPoseGenerator(minx, maxx, miny, maxy, minz, maxz)
        success = arm_control_client.move_arm_to_pose(test_pose)
        rospy.sleep(1.0)
