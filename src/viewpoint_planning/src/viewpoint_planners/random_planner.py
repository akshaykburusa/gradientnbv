import torch
import numpy as np

from scene_representation.voxel_grid import VoxelGrid
from viewpoint_planners.viewpoint_sampler import ViewpointSampler

from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import transform_from_rotation_translation


class RandomPlanner:
    """
    Class to plan random viewpoints
    """

    def __init__(
        self,
        start_pose: np.array,
        grid_size: np.array = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.array = np.array([0.002]),
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.random_params(start_pose, target_params)
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
        self.view_sampler = ViewpointSampler(num_samples)
        self.viewpoint = start_pose
        self.target_position = target_params
        self.rviz_visualizer = RvizVisualizer()

    def random_params(self, start_pose: np.array, target_params: np.array) -> None:
        """
        Initialize the parameters for random sampling
        """
        self.target_params = torch.tensor(
            target_params,
            dtype=torch.float32,
            device=self.device,
        )
        self.camera_bounds = np.array(
            [
                [
                    start_pose[0] - 0.2,
                    start_pose[1] - 0.1,
                    start_pose[2] - 0.15,
                    target_params[0] - 0.2,
                    target_params[1] - 0.2,
                    target_params[2] - 0.2,
                ],
                [
                    start_pose[0] + 0.2,
                    start_pose[1] + 0.1,
                    start_pose[2] + 0.15,
                    target_params[0] + 0.2,
                    target_params[1] + 0.2,
                    target_params[2] + 0.2,
                ],
            ]
        )

    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        :param depth_image: depth image (H, W)
        :param semantics: confidence scores and class ids (H, W, 2)
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

    def random_view(self) -> np.array:
        """
        Sample a viewpoint at random
        """
        view_samples = self.view_sampler.random_neighbour_sampler(
            self.viewpoint[:3],
            self.target_position,
            camera_limits=self.camera_bounds[:, :3],
            target_limits=self.camera_bounds[:, 3:],
        )
        self.rviz_visualizer.visualize_view_samples(
            numpy_to_pose_array(view_samples[:, :7])
        )
        random_index = np.random.randint(self.num_samples)
        viewpoint = view_samples[random_index, :7]
        self.target_position = view_samples[random_index, 7:]
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.viewpoint = viewpoint
        return self.viewpoint, 0.0, 1

    def visualize(self):
        """
        Visualize the voxel grid as a point cloud in rviz
        """
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        self.rviz_visualizer.visualize_camera_bounds(self.camera_bounds[:, :3])
