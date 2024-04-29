import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from scene_representation.voxel_grid import VoxelGrid

from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation


class GradientNBVPlanner(nn.Module):
    """
    Class to plan a locally optimal viewpoint using gradient-based optimization
    """

    def __init__(
        self,
        start_pose: np.array,
        grid_size: np.array = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.array = np.array([0.003]),
        grid_center: np.array = np.array([0.5, -0.4, 1.1]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array(
            [
                [685.5028076171875, 0.0, 485.35955810546875],
                [0.0, 685.6409912109375, 270.7330627441406],
                [0.0, 0.0, 1.0],
            ],
        ),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        target_params: np.array = np.array([0.5, -0.4, 1.1]),
    ) -> None:
        """
        Initialize the planner
        :param grid_size: size of the voxel grid in meters
        :param voxel_size: size of the voxels in meters
        :param grid_center: center of the voxel grid in meters
        :param image_size: size of the image in pixels
        :param num_pts_per_ray: number of points sampled per ray
        :param num_features: number of features per voxel
        """
        super(GradientNBVPlanner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.optimization_params(start_pose, target_params)
        self.voxel_grid = VoxelGrid(
            grid_size=grid_size,
            voxel_size=voxel_size,
            grid_center=grid_center,
            width=image_size[0],
            height=image_size[1],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            num_pts_per_ray=num_pts_per_ray,
            num_features=num_features,
            target_params=self.target_params,
            device=self.device,
        )
        self.num_samples = num_samples
        self.rviz_visualizer = RvizVisualizer()

    def optimization_params(
        self, start_pose: np.array, target_params: np.array
    ) -> None:
        """
        Initialize the optimization parameters
        """
        self.camera_params = nn.Parameter(
            torch.tensor(
                [
                    start_pose[0],
                    start_pose[1],
                    start_pose[2],
                    target_params[0],
                    target_params[1],
                    target_params[2],
                ],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        )
        self.target_params = torch.tensor(
            target_params,
            dtype=torch.float32,
            device=self.device,
        )
        self.camera_bounds = torch.tensor(
            [
                [
                    start_pose[0] - 0.2,
                    start_pose[1] - 0.1,
                    start_pose[2] - 0.15,
                    target_params[0] - 0.1,
                    target_params[1] - 0.1,
                    target_params[2] - 0.1,
                ],
                [
                    start_pose[0] + 0.2,
                    start_pose[1] + 0.1,
                    start_pose[2] + 0.15,
                    target_params[0] + 0.1,
                    target_params[1] + 0.1,
                    target_params[2] + 0.1,
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.03)

    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        :param depth_image: depth image (H x W)
        :param semantics: confidence scores and class ids (H x W x 2)
        :param viewpoint: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(
            viewpoint[3:], dtype=torch.float32, device=self.device
        )
        transform = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        return coverage

    def loss(self, target_pos: np.array) -> torch.tensor:
        """
        Compute the loss for the current viewpoint
        :return: loss
        """
        if target_pos is not None:
            self.target_params = torch.tensor(
                target_pos, dtype=torch.float32, device=self.device
            )
        else:
            self.target_params = self.camera_params[3:]
        loss, gain_image = self.voxel_grid.compute_gain(
            self.camera_params[:3], self.target_params
        )
        return loss, gain_image

    def next_best_view(self, target_pos=None) -> Tuple[np.array, float, int]:
        """
        Compute the next best viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        :return: loss
        :return: number of samples
        """
        for _ in range(self.num_samples):
            self.optimizer.zero_grad()
            loss, gain_image = self.loss(target_pos)
            loss.backward()
            self.optimizer.step()
            self.camera_params.data = torch.clamp(
                self.camera_params.data, self.camera_bounds[0], self.camera_bounds[1]
            )
        viewpoint = self.get_viewpoint()
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.rviz_visualizer.visualize_gain_image(gain_image)
        loss = loss.detach().cpu().numpy()
        return viewpoint, loss, self.num_samples

    def get_viewpoint(self) -> np.array:
        """
        Get the current viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        quat = look_at_rotation(self.camera_params[:3], self.camera_params[3:])
        quat = quat.detach().cpu().numpy()
        viewpoint = np.zeros(7)
        viewpoint[:3] = self.camera_params.detach().cpu().numpy()[:3]
        viewpoint[3:] = quat
        return viewpoint

    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        return voxel_points, sem_conf_scores, sem_class_ids

    def visualize(self):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        voxel_points, sem_conf_scores, sem_class_ids = self.get_occupied_points()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        # Visualize target
        # target = self.target_params.detach().cpu().numpy()
        target = self.camera_params.detach().cpu().numpy()[3:]
        rois = np.array([[*target, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        # Visualize camera bounds
        camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds)
