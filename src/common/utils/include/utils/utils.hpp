/*
 Author: Akshay Kumar Burusa
 Email: akshaykumar.burusa@wur.nl
 */

#ifndef UTILS_CPP
#define UTILS_CPP

#include <Eigen/Dense>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/ColorRGBA.h>

namespace utils {

inline geometry_msgs::Point point(float x, float y, float z) {
  geometry_msgs::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

inline geometry_msgs::Vector3 vector3(float x, float y, float z) {
  geometry_msgs::Vector3 vector3;
  vector3.x = x;
  vector3.y = y;
  vector3.z = z;
  return vector3;
}

inline geometry_msgs::Quaternion quaternion(float x, float y, float z,
                                            float w) {
  geometry_msgs::Quaternion quaternion;
  quaternion.x = x;
  quaternion.y = y;
  quaternion.z = z;
  quaternion.w = w;
  return quaternion;
}

inline geometry_msgs::Pose pose(geometry_msgs::Point point,
                                geometry_msgs::Quaternion quaternion) {
  geometry_msgs::Pose pose;
  pose.position = point;
  pose.orientation = quaternion;
  return pose;
}

inline std_msgs::ColorRGBA color_rgba(float r, float g, float b, float a) {
  std_msgs::ColorRGBA color_rgba;
  color_rgba.a = a;
  color_rgba.r = r;
  color_rgba.g = g;
  color_rgba.b = b;
  return color_rgba;
}

inline float norm(float x, float y, float z) {
  return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}

inline Eigen::Vector3d point_to_eigen(geometry_msgs::Point point) {
  Eigen::Vector3d vector3d;
  vector3d.x() = point.x;
  vector3d.y() = point.y;
  vector3d.z() = point.z;
  return vector3d;
}

inline geometry_msgs::Point eigen_to_point(Eigen::Vector3d vector3d) {
  geometry_msgs::Point point;
  point.x = vector3d.x();
  point.x = vector3d.x();
  point.x = vector3d.x();
  return point;
}

/* Create a matrix from translation and rotation.
 * Input: Translation, Rotation
 * Output: Transformation matrix */
inline Eigen::Matrix4d mat_from_transform(Eigen::Vector3d translation,
                                          Eigen::Quaterniond rotation) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
  transform.block<3, 1>(0, 3) = translation;
  return transform;
}

/* Find the Euclidean cost between two poses.
 * Input: Pose 1, Pose 2
 * Output: Euclidean distance */
inline float euclidean_distance(geometry_msgs::Pose pose_a,
                                geometry_msgs::Pose pose_b) {
  return norm(pose_a.position.x - pose_b.position.x,
              pose_a.position.y - pose_b.position.y,
              pose_a.position.z - pose_b.position.z);
}

inline float limit_angle(float angle) {
  angle = fmod(angle, 2 * M_PI);
  if (angle > M_PI)
    angle -= (2 * M_PI);
  else if (angle < -M_PI)
    angle += (2 * M_PI);
  return angle;
}

} // namespace utils

#endif // UTILS_CPP
