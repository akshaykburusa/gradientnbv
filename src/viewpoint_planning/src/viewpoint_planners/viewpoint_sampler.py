import numpy as np

from utils.py_utils import look_at_rotation


class ViewpointSampler:
    """
    Generate viewpoint samples for a given scene according to the given constraints
    """

    def __init__(self, num_samples: int = 1, seed: int = 2024):
        """
        Initialize the viewpoint sampler
        :param num_samples: Number of viewpoints to sample
        """
        self.num_samples = num_samples
        self.view_samples = np.empty((0, 7))
        self.seed = seed
        np.random.seed(self.seed)

    def sample(self, type: str = "random") -> np.ndarray:
        """
        Sample viewpoints for a given scene according to the given constraints
        :param type: Type of viewpoint sampler
        :return: PoseArray of sampled viewpoints
        """
        self.view_samples = np.empty((0, 7))
        # Call the appropriate sampler based on type
        if type == "random":
            self.random_sampler()
        elif type == "semi_cylindrical":
            self.semi_cylindrical_sampler()
        elif type == "five_dof":
            self.five_dof_sampler()
        elif type == "predefined":
            self.predefined_sampler()
        else:
            print("Invalid viewpoint sampler type. Using random sampler.")
            self.random_sampler()
        return self.view_samples

    def random_sampler(self) -> None:
        """
        Generate random viewpoints
        :return: None
        """
        bounds = np.array([[0.5, -0.15, 0.8], [0.6, 0.15, 1.5]])
        for _ in range(self.num_samples):
            position = np.random.uniform(bounds[0], bounds[1])
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
            pose = np.concatenate((position, orientation))
            self.view_samples = np.vstack((self.view_samples, pose))

    def semi_cylindrical_sampler(
        self,
        center: np.array = np.array([1.0, 0.0, 1.15]),
        length: float = 0.4,
        radius: float = 0.4,
        angle_limits: np.array = np.array([135.0, 225.0]),
    ) -> None:
        """
        Generate viewpoints on a semi-cylinder
        :param center: Center of the semi-cylinder
        :param length: Length of the semi-cylinder
        :param radius: Radius of the semi-cylinder
        :param angle_limits: Limits of the angle of the semi-cylinder
        :return: None
        """
        for _ in range(self.num_samples):
            # Randomly sample angle and height
            angle = np.random.uniform(angle_limits[0], angle_limits[1])
            height = np.random.uniform(center[2] - length / 2, center[2] + length / 2)
            # Convert polar coordinates to Cartesian coordinates
            x = center[0] + radius * np.cos(np.deg2rad(angle))
            y = center[1] + radius * np.sin(np.deg2rad(angle))
            z = height
            position = np.array([x, y, z])
            # Calculate the rotation quaternion to make the camera look at the center of the cylinder
            orientation = look_at_rotation(
                np.array([x, y, z]), np.array([center[0], center[1], z])
            )
            pose = np.concatenate((position, orientation))
            self.view_samples = np.vstack((self.view_samples, pose))

    def five_dof_sampler(
        self,
        camera_limits: np.array = np.array([[0.65, -0.4, 1.0], [0.8, 0.4, 1.3]]),
        target_limits: np.array = np.array([[1.0, 0.0, 1.15], [1.0, 0.0, 1.15]]),
    ) -> None:
        """
        Generate viewpoints such that the camera is looking at a target object
        :param camera_limits: Limits of the camera position
        :param target_limits: Limits of the target position
        :return: None
        """
        for _ in range(self.num_samples):
            # Randomly sample the camera and target positions
            camera_pos = np.random.uniform(camera_limits[0], camera_limits[1])
            target_pos = np.random.uniform(target_limits[0], target_limits[1])
            position = camera_pos
            # Calculate the rotation quaternion to make the camera look at the target
            orientation = look_at_rotation(camera_pos, target_pos)
            pose = np.concatenate((position, orientation))
            self.view_samples = np.vstack((self.view_samples, pose))

    def sample_five_dof(self, camera_limits, target_limits):
        self.view_samples = np.empty((0, 7))
        self.five_dof_sampler(camera_limits, target_limits)
        return self.view_samples

    def random_neighbour_sampler(
        self,
        current_position: np.array,
        current_target: np.array,
        camera_limits: np.array = np.array([[0.5, -0.35, 1.0], [0.7, 0.35, 1.3]]),
        target_limits: np.array = np.array([[0.85, -0.35, 1.0], [0.95, 0.35, 1.3]]),
    ) -> np.ndarray:
        """
        Generate camera and target positions
        :param current_position: Current camera position
        :param camera_limits: Limits of the camera position
        :param target_limits: Limits of the target position
        :return: None
        """
        view_samples = np.empty((0, 10))
        for _ in range(self.num_samples):
            # Sample a random point on a sphere of radius d
            d = 0.1
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            # Randomly sample the target position such that it is d meters away from the current target
            x = current_target[0] + d * np.cos(theta) * np.sin(phi)
            y = current_target[1] + d * np.sin(theta) * np.sin(phi)
            z = current_target[2] + d * np.cos(phi)
            target_pos = np.array([x, y, z])
            target_pos = np.clip(target_pos, target_limits[0], target_limits[1])
            # Sample a random point on a sphere of radius d
            d = 0.1
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            # Randomly sample the camera position such that it is d meters away from the current position
            x = current_position[0] + d * np.cos(theta) * np.sin(phi)
            y = current_position[1] + d * np.sin(theta) * np.sin(phi)
            z = current_position[2] + d * np.cos(phi)
            camera_pos = np.array([x, y, z])
            camera_pos = np.clip(camera_pos, camera_limits[0], camera_limits[1])
            position = camera_pos
            # Calculate the rotation quaternion to make the camera look at the target
            orientation = look_at_rotation(camera_pos, target_pos)
            pose = np.concatenate((position, orientation, target_pos))
            view_samples = np.vstack((view_samples, pose))
        return view_samples

    def predefine_start_pose(
        self,
        target_position: np.array = np.array([0.5, -0.4, 1.0]),
        distance: float = 0.35,
    ) -> np.ndarray:
        """
        Predefine the start pose of the camera
        :param center: Center of the target object
        :param distance: Distance of the camera from the target object
        :return: Start pose of the camera
        """
        position = np.array(
            [target_position[0], target_position[1] + distance, target_position[2]]
        )
        orientation = look_at_rotation(position, target_position)
        pose = np.concatenate((position, orientation))
        return pose
