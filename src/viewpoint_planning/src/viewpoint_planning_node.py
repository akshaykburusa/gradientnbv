#!/usr/bin/env python3
# ROS node to run the viewpoint planning algorithms

import rospy
from viewpoint_planners.viewpoint_planning import ViewpointPlanning

if __name__ == "__main__":
    rospy.init_node("viewpoint_planning")
    viewpoint_planner = ViewpointPlanning()

    while not rospy.is_shutdown():
        viewpoint_planner.run()
