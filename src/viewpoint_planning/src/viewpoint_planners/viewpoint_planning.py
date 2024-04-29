import rospy
import numpy as np

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from viewpoint_planners.gradientnbv_planner import GradientNBVPlanner
from viewpoint_planners.random_planner import RandomPlanner

from utils.sdf_spawner import SDFSpawner
from utils.py_utils import numpy_to_pose


class ViewpointPlanning:
    def __init__(self):
        self.arm_control = ArmControlClient()
        self.perceiver = Perceiver()
        self.viewpoint_sampler = ViewpointSampler()
        self.sdf_spawner = SDFSpawner()
        self.config()
        # Gradient-based planner
        self.gradient_planner = GradientNBVPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=1,
        )
        # Random planner
        self.random_planner = RandomPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=1,
        )

    def run(self):
        self.run_gradient_nbv()
        # self.run_random()

    def config(self):
        # Configure target
        self.target_position = np.array([0.5, -0.4, 1.1])
        # occlusion_position = np.array([0.5, -0.3, 1.25])  # top occlusion
        # occlusion_position = np.array([0.5, -0.3, 0.95])  # bottom occlusion
        occlusion_position = np.array([0.65, -0.3, 1.1])  # left occlusion
        # occlusion_position = np.array([0.35, -0.3, 1.1])  # right occlusion
        self.sdf_spawner.spawn_box(occlusion_position)
        # Configure initial camera viewpoint
        self.camera_pose = self.viewpoint_sampler.predefine_start_pose(
            self.target_position
        )
        self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        # Configure scene
        self.grid_size = np.array([0.3, 0.6, 0.3])
        self.grid_center = self.target_position
        # Configure camera
        camera_info = self.perceiver.get_camera_info()
        self.image_size = np.array([camera_info.width, camera_info.height])
        self.intrinsics = np.array(camera_info.K).reshape(3, 3)

    def run_gradient_nbv(self):
        self.camera_pose, loss, iters = self.gradient_planner.next_best_view()
        is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        rospy.sleep(1.0)
        if is_success:
            depth_image, points, semantics = self.perceiver.run()
            coverage = self.gradient_planner.update_voxel_grid(
                depth_image, semantics, self.camera_pose
            )
            print("Target coverage: ", coverage)
            self.gradient_planner.visualize()

    def run_random(self):
        self.camera_pose, _, _ = self.random_planner.random_view()
        is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        rospy.sleep(1.0)
        if is_success:
            depth_image, points, semantics = self.perceiver.run()
            coverage = self.random_planner.update_voxel_grid(
                depth_image, semantics, self.camera_pose
            )
            print("Target coverage: ", coverage)
            self.random_planner.visualize()
