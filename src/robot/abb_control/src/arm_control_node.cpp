/*
 Author: Akshay Kumar Burusa
 Email: akshaykumar.burusa@wur.nl
 */

#include <abb_control/arm_control.hpp>
#include <utils/utils.hpp>

int main(int argc, char **argv) {
  ros::init(argc, argv, "arm_control");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(2);
  spinner.start();

  // ROS parameters.
  std::string world_frame_id;
  nh.getParam("/agent/frames/world_frame_id", world_frame_id);
  std::string arm_name;
  XmlRpc::XmlRpcValue obstacles;
  std::vector<geometry_msgs::Vector3> dims;
  geometry_msgs::PoseArray obstacle_poses;
  obstacle_poses.header.frame_id = world_frame_id;
  nh.getParam("arm_control/arm_name", arm_name);
  nh.getParam("obstacles", obstacles);
  if (obstacles.getType() == XmlRpc::XmlRpcValue::TypeArray) {
    for (int i = 0; i < obstacles.size(); i++) {
      XmlRpc::XmlRpcValue obstacle = obstacles[i];
      geometry_msgs::Vector3 dim;
      dim = utils::vector3(static_cast<double>(obstacle[0]),
                           static_cast<double>(obstacle[1]),
                           static_cast<double>(obstacle[2]));
      dims.push_back(dim);
      geometry_msgs::Pose obstacle_pose;
      obstacle_pose =
          utils::pose(utils::point(static_cast<double>(obstacle[3]),
                                   static_cast<double>(obstacle[4]),
                                   static_cast<double>(obstacle[5])),
                      utils::quaternion(static_cast<double>(obstacle[6]),
                                        static_cast<double>(obstacle[7]),
                                        static_cast<double>(obstacle[8]),
                                        static_cast<double>(obstacle[9])));
      obstacle_poses.poses.push_back(obstacle_pose);
    }
  }

  // Object instantiations.
  arm_control::ArmControl arm_control(arm_name);

  arm_control.AddCollisionObjects(dims, obstacle_poses);

  ros::ServiceServer move_arm_to_pose_srv = nh.advertiseService(
      "move_arm_to_pose", &arm_control::ArmControl::MoveToGoalSrv,
      &arm_control);

  ros::waitForShutdown();
  return 0;
}