"""
Author: Akshay K. Burusa
Maintainer: Akshay K. Burusa
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from scene_representation.raysampler import RaySampler
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation


class VoxelGrid:
    """
    3D representation to store occupancy information and other features (e.g. semantics) over
    multiple viewpoints
    """

    def __init__(
        self,
        grid_size: torch.tensor,
        voxel_size: torch.tensor,
        grid_center: torch.tensor,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        z_near: float = 0.05,
        z_far: float = 0.6,
        target_params: torch.tensor = None,
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        eps: torch.float32 = 1e-7,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """
        Constructor
        :param grid_size: size of the voxel grid
        :param voxel_size: size of each voxel
        :param grid_center: center of the voxel grid
        :param width: image width
        :param height: image height
        :param fx: focal length along x-axis
        :param fy: focal length along y-axis
        :param cx: principal point along x-axis
        :param cy: principal point along y-axis
        :param z_near: near clipping plane
        :param z_far: far clipping plane
        :param num_pts_per_ray: number of points sampled along each ray
        :param eps: epsilon value for numerical stability
        :param device: device to use for computation
        """
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.grid_center = grid_center
        self.width = width
        self.height = height
        self.num_pts_per_ray = num_pts_per_ray
        self.num_features = num_features
        self.eps = eps
        self.device = device

        self.voxel_dims = (grid_size / voxel_size).long()
        self.origin = grid_center - grid_size / 2.0
        self.min_bound = self.origin
        self.max_bound = self.origin + grid_size

        # 4D voxel grid
        self.voxel_grid = torch.zeros(
            (
                self.voxel_dims[0],
                self.voxel_dims[1],
                self.voxel_dims[2],
                num_features,  # ROIs, occ_prob, sem_conf, sem_class
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.voxel_grid[..., 1] = 0.5  # Initialize occupancy probability as 0.5
        self.voxel_grid[..., 2] = self.eps  # Initialize semantic confidence close to 0
        self.voxel_grid[..., 3] = -1  # Initialize semantic class to background
        # Define regions of interest around the target
        self.set_target_roi(target_params)

        # Occupancy and semantic information along a ray
        # Occupancy probabilities along the ray is initialized to 0.2, which gives a log odds of -1.4
        ray_occ = -0.4 * torch.ones(
            (self.num_pts_per_ray, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.ray_occ = ray_occ.unsqueeze(0).repeat(
            width * height, 1, 1
        )  # (W x H, num_pts_per_ray, 1)
        # Semantic confidence along the ray is initialized to 0.2, which gives a log odds of -1.4
        ray_sem_conf = -0.4 * torch.ones(
            num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )
        # Semantic class along the ray is initialized to -1 (background)
        ray_sem_cls = -1 * torch.ones(
            num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )
        ray_sem = torch.stack((ray_sem_conf, ray_sem_cls), dim=-1)
        self.ray_sem = ray_sem.unsqueeze(0).repeat(
            width * height, 1, 1
        )  # (W x H, num_pts_per_ray, 2)

        # Ray sampler
        self.ray_sampler = RaySampler(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            z_near=z_near,
            z_far=z_far,
            device=device,
        )
        self.t_vals = torch.linspace(
            0.0,
            1.0,
            self.num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )

    def insert_depth_and_semantics(
        self,
        depth_image: torch.tensor,
        semantics: torch.tensor,
        transforms: torch.tensor,
    ) -> None:
        """
        Insert a point cloud into the voxel grid
        :param depth_image: depth image from the current viewpoint (W x H)
        :param semantics: semantic confidences and labels from the current viewpoint (2 x W x H)
        :param position: position of the current viewpoint (3,)
        :param orientation: orientation of the current viewpoint (4,)
        :return: None
        """
        # Convert depth image to point cloud
        (
            ray_origins,
            ray_directions,
            points_mask,
        ) = self.ray_sampler.ray_origins_directions(
            depth_image=depth_image, transforms=transforms
        )
        ray_points = (
            ray_directions[:, :, None, :] * self.t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)

        # # Visualize ray points in Open3D
        # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(ray_points.detach().cpu().numpy())
        # o3d.visualization.draw_geometries([origin_frame, pcd])

        # Convert point cloud to voxel grid coordinates
        grid_coords = torch.div(
            ray_points - self.origin, self.voxel_size, rounding_mode="floor"
        )
        valid_indices = self.get_valid_indices(grid_coords, self.voxel_dims)
        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)
        # Get the log odds of the occupancy and semantic probabilities
        log_odds = torch.log(
            torch.div(
                self.voxel_grid[gx, gy, gz, 1:3], 1.0 - self.voxel_grid[gx, gy, gz, 1:3]
            )
        )
        # Update the log odds of the occupancy probabilities
        ray_occ = self.ray_occ.clone()
        ray_occ[:, -2:, :] = points_mask.permute(1, 0).repeat(1, 2).unsqueeze(-1)
        log_odds[..., 0] += ray_occ.view(-1, 1)[valid_indices, -1]
        # Update the log odds of the semantic probabilities
        ray_sem = self.ray_sem.clone()
        ray_sem[..., -1, :] = semantics.view(-1, 2)
        ray_sem = ray_sem.view(-1, 2)
        log_odds[..., 1] += ray_sem[valid_indices, 0]
        # Convert the log odds back to occupancy and semantic probabilities
        odds = torch.exp(log_odds)
        self.voxel_grid[gx, gy, gz, 1:3] = torch.div(odds, 1.0 + odds)
        self.voxel_grid[..., 1:3] = torch.clamp(
            self.voxel_grid[..., 1:3], self.eps, 1.0 - self.eps
        )
        self.voxel_grid[gx, gy, gz, 3] = ray_sem[valid_indices, 1]
        # Check the values within the target bounds and count the number of updated voxels
        if self.target_bounds is not None:
            target_voxels = self.voxel_grid[
                self.target_bounds[0] : self.target_bounds[3],
                self.target_bounds[1] : self.target_bounds[4],
                self.target_bounds[2] : self.target_bounds[5],
                2,
            ]
            coverage = torch.sum((target_voxels != 0.5)) / target_voxels.numel() * 100
            return coverage

    def compute_gain(
        self,
        camera_params: torch.tensor,
        target_params: torch.tensor,
    ) -> torch.tensor:
        """
        Compute the gain for a given set of parameters
        :param camera_params: camera parameters
        :param target_params: target parameters
        :param current_params: current parameters
        :return: total gain for the viewpoint defined by the parameters
        """
        quat = look_at_rotation(camera_params, target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_params[None, :]
        )
        # Compute point cloud by ray-tracing along ray origins and directions
        t_vals = self.t_vals.clone().requires_grad_()
        ray_origins, ray_directions, _ = self.ray_sampler.ray_origins_directions(
            transforms=transforms
        )
        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)
        ray_points_nor = self.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        # Sample the occupancy probabilities and semantic confidences along each ray
        grid = self.voxel_grid[None, ..., 1:3].permute(4, 0, 1, 2, 3)
        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs = occ_sem_confs.view(2, -1, self.num_pts_per_ray)
        occ_sem_confs = occ_sem_confs.clamp(self.eps, 1.0 - self.eps)
        # Compute the entropy of the semantic confidences along each ray
        opacities = torch.sigmoid(1e7 * (occ_sem_confs[0, ...] - 0.51))
        transmittance = self.shifted_cumprod(1.0 - opacities)
        ray_gains = transmittance * self.entropy(occ_sem_confs[1, ...])
        # Create a gain image for visualization
        gain_image = ray_gains.view(-1, self.num_pts_per_ray).sum(1)
        gain_image = gain_image.view(self.height, self.width)
        gain_image = gain_image - gain_image.min()
        # gain_image = gain_image / gain_image.max()
        gain_image = gain_image / 32.0
        gain_image = gain_image.detach().cpu().numpy()
        gain_image = plt.cm.viridis(gain_image)[..., :3]
        # Compute the semantic gain
        semantic_gain = torch.log(torch.mean(ray_gains) + self.eps)
        loss = -semantic_gain
        return loss, gain_image

    def entropy(self, probs: torch.tensor) -> torch.tensor:
        """
        Compute the entropy of a set of probabilities
        :param probs: tensor of probabilities
        :return: tensor of entropies
        """
        probs_inv = 1.0 - probs
        gains = -(probs * torch.log2(probs)) - (probs_inv * torch.log2(probs_inv))
        return gains

    def set_target_roi(self, target_params: torch.tensor) -> None:
        # Define regions of interest around the target
        self.target_bounds = None
        if target_params is None:
            return
        target_coords = torch.div(
            target_params - self.origin, self.voxel_size, rounding_mode="floor"
        ).to(torch.long)
        x_min = torch.clamp(target_coords[0] - 25, 0, self.voxel_dims[0])
        x_max = torch.clamp(target_coords[0] + 25, 0, self.voxel_dims[0])
        y_min = torch.clamp(target_coords[1] - 25, 0, self.voxel_dims[1])
        y_max = torch.clamp(target_coords[1] + 25, 0, self.voxel_dims[1])
        z_min = torch.clamp(target_coords[2] - 25, 0, self.voxel_dims[2])
        z_max = torch.clamp(target_coords[2] + 25, 0, self.voxel_dims[2])
        self.voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max, 0] = 1
        self.voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max, 2] = 0.5
        self.target_bounds = torch.tensor(
            [x_min, y_min, z_min, x_max, y_max, z_max], device=self.device
        )
        self.voxel_grid[..., 1:3] = torch.clamp(
            self.voxel_grid[..., 1:3], self.eps, 1.0 - self.eps
        )

    def get_valid_indices(
        self, grid_coords: torch.tensor, dims: torch.tensor
    ) -> torch.tensor:
        """
        Get the indices of the grid coordinates that are within the grid bounds
        :param grid_coords: tensor of grid coordinates
        :param dims: tensor of grid dimensions
        :return: tensor of valid indices
        """
        valid_indices = (
            (grid_coords[:, 0] >= 0)
            & (grid_coords[:, 0] < dims[0])
            & (grid_coords[:, 1] >= 0)
            & (grid_coords[:, 1] < dims[1])
            & (grid_coords[:, 2] >= 0)
            & (grid_coords[:, 2] < dims[2])
        )
        return valid_indices

    def normalize_3d_coordinate(self, points):
        """
        Normalize a tensor of 3D points to the range [-1, 1] along each axis.
        :param points: tensor of 3D points of shape (N, 3)
        :return: tensor of normalized 3D points of shape (N, 3)
        """
        # Compute the range of values for each dimension
        x_min, y_min, z_min = self.min_bound
        x_max, y_max, z_max = self.max_bound
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        # Normalize the points to the range [-1, 1]
        n_points = points.clone()
        n_points_out = torch.zeros_like(n_points)
        n_points_out[..., 0] = 2.0 * (n_points[..., 2] - z_min) / z_range - 1.0
        n_points_out[..., 1] = 2.0 * (n_points[..., 1] - y_min) / y_range - 1.0
        n_points_out[..., 2] = 2.0 * (n_points[..., 0] - x_min) / x_range - 1.0
        return n_points_out

    def shifted_cumprod(self, x: torch.tensor, shift: int = 1) -> torch.tensor:
        """
        Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of ones and removes
        `shift` trailing elements to/from the last dimension of the result
        :param x: tensor of shape (N, ..., C)
        :param shift: number of elements to prepend/remove
        :return: tensor of shape (N, ..., C)
        """
        x_cumprod = torch.cumprod(x, dim=-1)
        x_cumprod_shift = torch.cat(
            [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
        )
        return x_cumprod_shift

    def get_occupied_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        grid_coords = torch.nonzero(self.voxel_grid[..., 1] > 0.5)
        semantics = self.voxel_grid[
            grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 2
        ]
        class_ids = self.voxel_grid[
            grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 3
        ]
        points = grid_coords * self.voxel_size + self.origin
        return points, semantics, class_ids
