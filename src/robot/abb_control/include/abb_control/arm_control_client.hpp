/*
 Author: Akshay Kumar Burusa
 Email: akshaykumar.burusa@wur.nl
 */

#ifndef ARM_CONTROL_CLIENT_HPP
#define ARM_CONTROL_CLIENT_HPP

#include <ros/ros.h>
#include <random>
#include <geometry_msgs/Pose.h>

#include <abb_control/ArmGoal.h>

namespace arm_control_client
{
class ArmControlClient
{
protected:
  // ROS service.
  abb_control::ArmGoal srv_;
  ros::ServiceClient client_;
  ros::NodeHandle nh_;
  // Workspace borders.
  float minx_;
  float maxx_;
  float miny_;
  float maxy_;
  float minz_;
  float maxz_;

public:
  /* Constructor.
   * Input: ROS nodehandle, workspace constraints in x,y,z axes. */
  ArmControlClient(const ros::NodeHandle&, float minx = 0.40, float maxx = 0.80, float miny = -0.40, float maxy = 0.40,
                   float minz = 1.00, float maxz = 1.40);
  // Destructor.
  ~ArmControlClient();
  /* Random pose generator.
   * Output: A randomly generator end-effector goal pose. */
  geometry_msgs::Pose GetRandomPose();
  /* Call arm control service and move arm to goal pose.
   * Input: Goal pose for the end-effector.
   * Output: (boolean) Success. */
  bool MoveArmToGoal(geometry_msgs::Pose);
};
}  // namespace arm_control_client

#endif  // ARM_CONTROL_CLIENT_HPP
