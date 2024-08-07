import numpy as np
from gymnasium.spaces import Box, Dict
from collections import OrderedDict
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped
import cv2
from rclpy.qos import qos_profile_sensor_data
from frontiers import get_frontiers, paint_frontiers


class Observation:
    def __init__(self, grid_size, num_envs, drone_interface_list):
        self.grid_size = grid_size
        self.observation_space = Dict(
            {
                "image": Box(
                    low=0, high=2, shape=(1, grid_size, grid_size), dtype=np.uint8
                ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown
                "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                # Position of the drone in the grid
            }
        )

        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((num_envs,) +
                 tuple(shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.drone_interface_list = drone_interface_list
        # Make the necessary subscription and variables in order to get the occupancy map
        self.grid_map_sub = self.drone_interface_list[0].create_subscription(
            GridMap, "/map_server/grid_map", self.grid_map_callback, qos_profile_sensor_data
        )
        self.grid_matrix = np.zeros((1, grid_size, grid_size), dtype=np.uint8)

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def _save_obs(self, obs: VecEnvObs, env_id):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_id] = obs
            else:
                self.buf_obs[key][env_id] = obs[key]

    # TODO: Return observation for a specific environment
    def _get_obs(self, env_id) -> VecEnvObs:
        position = self.convert_pose_to_grid_position(self.drone_interface_list[env_id].position)
        obs = {"image": self.grid_matrix, "position": position}
        return obs

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return np.array([x, y], dtype=np.int32)

    def grid_map_callback(self, msg: GridMap):
        # Get the grid map from the message and save it in the grid_matrix variable
        data = msg.data
        # Initialize the matrix with zeros
        matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        all_data = []
        for row in data:
            all_data.extend(row.data)
        # Flatten the data
        flat_data = np.array(all_data, dtype=np.float32)

        matrix = flat_data.reshape((self.grid_size, self.grid_size))
        matrix = matrix.swapaxes(0, 1)
        # Handle NaN values: convert NaNs to a specific value (2 for unknown)
        matrix[np.isnan(matrix)] = 2
        # Handle other values: convert all non-zero values to (1 for occupied)
        matrix[(matrix != 0) & (matrix != 2)] = 1

        # Convert to uint8 and reshape
        matrix = matrix.astype(np.uint8)
        self.grid_matrix = matrix[np.newaxis, :, :]  # Add batch dimension
        # if you want to show grid uncomment the following line

    def show_image_with_frontiers(self):
        image = self.process_image(self.grid_matrix)
        centroids, frontiers = get_frontiers(image)
        new_img = paint_frontiers(image, frontiers, centroids)
        cv2.imshow('frontiers', new_img)
        cv2.waitKey(1)

    def save_image(self, path: str):
        image = self.process_image(self.grid_matrix)

        # Save the image using OpenCV
        cv2.imwrite(path, image)
        cv2.waitKey(1)

    def process_image(self, image_matrix: np.ndarray):
        image_matrix = image_matrix[0]
        image = np.zeros((image_matrix.shape[0], image_matrix.shape[1], 1), dtype=np.uint8)

        color_map = {
            0: [255],  # White
            1: [0],       # Black
            2: [128]  # Grey
        }

        # Map the matrix values to the corresponding colors
        for i in range(image_matrix.shape[1]):
            for j in range(image_matrix.shape[0]):
                image[i, j] = color_map[image_matrix[i, j]]

        return image
