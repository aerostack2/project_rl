import numpy as np
import rclpy
from gymnasium.spaces import Box, Dict
from collections import OrderedDict
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
import cv2
from rclpy.qos import qos_profile_sensor_data
from as2_msgs.srv import GetFrontiers
from geometry_msgs.msg import PoseStamped
# from as2_msgs.msg import GetFrontierReq
# from as2_msgs.msg import GetFrontierRes
from frontiers import get_frontiers, paint_frontiers
import time


# @dataclass
# class ObservationMultiInputPolicy:
#     grid_size: int
#     observation_space = Dict(
#         {
#             "image": Box(
#                 low=0, high=255, shape=(1, grid_size, grid_size), dtype=np.uint8
#             ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown, 3: frontier point
#             "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
#             # Position of the drone in the grid
#         }
#     )


# @dataclass
# class ObservationMlpPolicy:
#     grid_size: int
#     observation_space = Box(low=0, high=1, shape=(grid_size * grid_size + 2,), dtype=np.float32)


class Observation:
    def __init__(self, grid_size, num_envs: int, drone_interface_list, policy_type: str):
        self.grid_size = grid_size

        if policy_type == "MlpPolicy":
            self.observation_space = Box(low=0, high=1, shape=(
                grid_size * grid_size + 2,), dtype=np.float32)
        elif policy_type == "MultiInputPolicy":
            self.observation_space = Dict(
                {
                    "image": Box(
                        low=0, high=255, shape=(1, grid_size, grid_size), dtype=np.uint8
                    ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown, 3: frontier point
                    "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                    # Position of the drone in the grid
                }
            )

        # self.observation_space = Dict(
        #     {
        #         "image": Box(
        #             low=0, high=255, shape=(1, grid_size, grid_size), dtype=np.uint8
        #         ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown, 3: frontier point
        #         "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
        #         # Position of the drone in the grid
        #     }
        # )
        self.policy_type = policy_type

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
            OccupancyGrid, "/map_server/map_filtered", self.grid_map_callback, qos_profile_sensor_data
        )

        self.get_frontiers_srv = self.drone_interface_list[0].create_client(
            GetFrontiers, "/get_frontiers"
        )

        # self.get_frontiers_pub = self.drone_interface_list[0].create_publisher(
        #     GetFrontierReq, "/get_frontiers_req", 1
        # )

        # self.get_frontiers_sub = self.drone_interface_list[0].create_subscription(
        #     GetFrontierRes, "/get_frontiers_res", self.get_frontiers_callback, 1
        # )

        self.grid_matrix = np.zeros((1, grid_size, grid_size), dtype=np.uint8)

        self.frontiers = []  # List of frontiers by coordinates in earth
        self.position_frontiers = []  # List of frontiers by coordinates in grid
        self.chosen_frontiers = []  # List of frontiers chosen by the drones
        self.wait_for_map = 1
        self.wait_for_frontiers = 0

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def _save_obs(self, obs, env_id):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_id] = obs
            else:
                self.buf_obs[key][env_id] = obs[key]

    def _get_obs(self, env_id):
        position = self.convert_pose_to_grid_position(self.drone_interface_list[env_id].position)
        self.put_frontiers_in_grid()
        # self.put_drone_in_grid(env_id)
        # self.show_image_with_frontiers()
        # self.save_image_as_csv("frontiers.csv")
        self.save_image_as_txt("frontiers.txt")
        if self.policy_type == "MlpPolicy":
            obs = self.grid_matrix.flatten()
            obs = np.append(obs, position)
        elif self.policy_type == "MultiInputPolicy":
            obs = {"image": self.grid_matrix, "position": position}
        return obs

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return (np.array([x, y], dtype=np.int32))

    def grid_map_callback(self, msg: OccupancyGrid):
        # Get the grid map from the message and save it in the grid_matrix variable
        print("entra aqui")
        if len(msg.data) == 0:
            return
        if self.wait_for_map == 1:
            return
        # Initialize the matrix with zeros
        matrix = np.array(msg.data, dtype=np.float32).reshape((self.grid_size, self.grid_size))
        matrix = matrix.swapaxes(0, 1)
        matrix = np.rot90(matrix, k=1, axes=(0, 1))
        matrix = np.rot90(matrix, k=1, axes=(0, 1))
        # Handle other values: convert all non-zero values to (1 for occupied)
        matrix[(matrix != -1) & (matrix != 0)] = 1 / 3 * 255
        # Handle NaN values: convert NaNs to a specific value (2 for unknown)
        matrix[matrix == -1] = 2 / 3 * 255
        # Convert to uint8 and reshape
        matrix = matrix.astype(np.uint8)
        self.grid_matrix = matrix[np.newaxis, :, :]  # Add batch dimension
        self.wait_for_map = 1

    def put_frontiers_in_grid(self):
        offsets = [
            (0, 0), (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]
        for frontier in self.frontiers:
            frontier_position = self.convert_pose_to_grid_position(frontier)
            # paint a square around the frontier

            for dx, dy in offsets:
                new_x, new_y = frontier_position[0] + dx, frontier_position[1] + dy
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:  # Ensure within bounds
                    self.grid_matrix[0, new_y, new_x] = 3 / 3 * 255

    # def put_drone_in_grid(self, env_idx):
    #     drone_position = self.convert_pose_to_grid_position(
    #         self.drone_interface_list[env_idx].position)
    #     self.grid_matrix[0, drone_position[1], drone_position[0]] = 4 / 4 * 255

    def show_image_with_frontiers(self):
        image = self.process_image(self.grid_matrix)
        # centroids, frontiers = get_frontiers(image)
        # new_img = paint_frontiers(image, frontiers, centroids)
        cv2.imshow('frontiers', image)
        cv2.waitKey(10)

    def save_image_as_txt(self, path: str):
        # image = self.process_image(self.grid_matrix)
        image = self.grid_matrix[0]
        print(image.shape)
        with open(path, 'w') as f:
            for row in image:
                for cell in row:
                    f.write(f"{cell} ")
                f.write("\n")

    def save_image_as_csv(self, path: str):
        image = self.grid_matrix[0]
        np.savetxt(path, image, delimiter=",")

    def order_and_get_frontiers_and_position(self, env_id):
        # Call the service to get the frontiers

        get_frontiers_req = GetFrontiers.Request()
        get_frontiers_req.explorer_id = f"drone{env_id}"
        get_frontiers_res = self.get_frontiers_srv.call(get_frontiers_req)
        self.frontiers = []
        self.position_frontiers = []
        for frontier in get_frontiers_res.frontiers:
            self.frontiers.append([frontier.point.x, frontier.point.y])
        self.frontiers = self.order_frontiers(self.frontiers)
        for frontier in self.frontiers:
            self.position_frontiers.append(self.convert_pose_to_grid_position(frontier))
        return self.frontiers, self.position_frontiers

    # def call_get_frontiers_with_msg(self, env_id):
    #     get_frontiers_req = GetFrontierReq()
    #     get_frontiers_req.explorer_id = f"drone{env_id}"
    #     self.get_frontiers_pub.publish(get_frontiers_req)
    #     self.wait_for_frontiers = 0
    #     self.wait_for_map = 0

    def get_frontiers_and_position_with_msg(self, env_id):
        return self.frontiers, self.position_frontiers

    def get_frontiers_and_position(self, env_id):
        get_frontiers_req = GetFrontiers.Request()
        get_frontiers_req.explorer_id = f"drone{env_id}"
        get_frontiers_res = self.get_frontiers_srv.call(get_frontiers_req)
        self.frontiers = []
        self.position_frontiers = []
        for frontier in get_frontiers_res.frontiers:
            self.frontiers.append([frontier.point.x, frontier.point.y])
            self.position_frontiers.append(self.convert_pose_to_grid_position([
                                           frontier.point.x, frontier.point.y]))
        return self.frontiers, self.position_frontiers

    def order_frontiers(self, frontiers):
        # Order frontiers from left to right and top to bottom
        sorted_frontiers = sorted(frontiers, key=lambda point: (-point[1], -point[0]))
        return sorted_frontiers

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
            1: [0],
            2: [128],  # Grey
            3: [50]
        }

        # Map the matrix values to the corresponding colors
        for i in range(image_matrix.shape[1]):
            for j in range(image_matrix.shape[0]):
                image[i, j] = color_map[image_matrix[i, j]]

        return image

    # def get_frontiers_callback(self, response: GetFrontierRes):
    #     self.frontiers = []
    #     for frontier in response.frontiers:
    #         self.frontiers.append([frontier.point.x, frontier.point.y])
    #         self.position_frontiers.append(self.convert_pose_to_grid_position([
    #                                        frontier.point.x, frontier.point.y]))
    #     self.wait_for_frontiers = 1


# class ObservationDiscrete:
#     def __init__(self, grid_size, num_envs: int, drone_interface_list, policy_type: str):
#         self.grid_size = grid_size
#         if policy_type == "MlpPolicy":
#             self.observation_space = Box

class ObservationAsync:

    FREE = np.uint8(0)
    OCCUPIED = np.uint8(1 / 4 * 255)
    UNKNOWN = np.uint8(2 / 4 * 255)
    FRONTIER = np.uint8(3 / 4 * 255)
    DRONE = np.uint8(4 / 4 * 255)

    def __init__(self, grid_size, num_envs: int, num_drones: int, env_idx: int, drone_interface_list, policy_type: str):
        self.grid_size = grid_size

        if policy_type == "MlpPolicy":
            self.observation_space = Box(low=0, high=1, shape=(
                grid_size * grid_size + 2,), dtype=np.float32)
        elif policy_type == "MultiInputPolicy":
            self.observation_space = Dict(
                {
                    "image": Box(
                        low=0, high=255, shape=(1, grid_size, grid_size), dtype=np.uint8
                    ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown, 3: frontier point
                    "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                    # Position of the drone in the grid
                }
            )

        # self.observation_space = Dict(
        #     {
        #         "image": Box(
        #             low=0, high=255, shape=(1, grid_size, grid_size), dtype=np.uint8
        #         ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown, 3: frontier point
        #         "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
        #         # Position of the drone in the grid
        #     }
        # )
        self.policy_type = policy_type

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
            OccupancyGrid, "/map_server/map_filtered", self.grid_map_callback, qos_profile_sensor_data
        )

        self.get_frontiers_srv = self.drone_interface_list[0].create_client(
            GetFrontiers, "/get_frontiers"
        )

        self.swarm_position_sub = []
        self.swarm_position = {}

        for i in range(num_drones):
            if f"drone{i}" != self.drone_interface_list[0].drone_id:
                self.swarm_position_sub.append(self.drone_interface_list[0].create_subscription(
                    PoseStamped, f"/drone{i}/self_localization/pose", getattr(
                        self, f"swarm_position_callback_{i}"), qos_profile_sensor_data
                ))
            self.swarm_position[f"drone{i}"] = (0, 0)

            # self.get_frontiers_pub = self.drone_interface_list[0].create_publisher(
            #     GetFrontierReq, "/get_frontiers_req", 1
            # )

            # self.get_frontiers_sub = self.drone_interface_list[0].create_subscription(
            #     GetFrontierRes, "/get_frontiers_res", self.get_frontiers_callback, 1
            # )

        self.grid_matrix = np.zeros((1, grid_size, grid_size), dtype=np.uint8)

        self.frontiers = []  # List of frontiers by coordinates in earth
        self.position_frontiers = []  # List of frontiers by coordinates in grid
        self.chosen_frontiers = []  # List of frontiers chosen by the drones
        self.wait_for_map = 0
        self.wait_for_frontiers = 0
        self.env_idx = env_idx

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def _save_obs(self, obs, env_id):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_id] = obs
            else:
                self.buf_obs[key][env_id] = obs[key]

    def _get_obs(self, env_id):
        position = self.convert_pose_to_grid_position(self.drone_interface_list[env_id].position)
        print("Drone ", self.env_idx, " position: ", position)
        self.put_frontiers_in_grid()
        print("Drone ", self.env_idx, " frontiers in grid")
        self.put_other_drones_in_grid()
        print("Drone ", self.env_idx, " drones in grid")
        # self.save_image_as_csv("frontiers.csv")
        # self.show_image_with_frontiers()
        if self.policy_type == "MlpPolicy":
            obs = self.grid_matrix.flatten()
            obs = np.append(obs, position)
        elif self.policy_type == "MultiInputPolicy":
            obs = {"image": self.grid_matrix, "position": position}
        self.wait_for_map = 0
        return obs

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return (int(x), int(y))

    def grid_map_callback(self, msg: OccupancyGrid):
        # Get the grid map from the message and save it in the grid_matrix variable
        if len(msg.data) == 0:
            return
        if self.wait_for_map == 1:
            return
        # Initialize the matrix with zeros
        matrix = np.array(msg.data, dtype=np.float32).reshape((self.grid_size, self.grid_size))
        matrix = matrix.swapaxes(0, 1)
        matrix = np.rot90(matrix, k=1, axes=(0, 1))
        matrix = np.rot90(matrix, k=1, axes=(0, 1))
        # Handle other values: convert all non-zero values to (1 for occupied)
        matrix[(matrix != -1) & (matrix != 0)] = self.OCCUPIED
        # Handle NaN values: convert NaNs to a specific value (2 for unknown)
        matrix[matrix == -1] = self.UNKNOWN
        # Convert to uint8 and reshape
        matrix = matrix.astype(np.uint8)
        self.grid_matrix = matrix[np.newaxis, :, :]  # Add batch dimension
        self.wait_for_map = 1

    def swarm_position_callback_0(self, msg: PoseStamped):
        self.swarm_position["drone0"] = self.convert_pose_to_grid_position(
            [msg.pose.position.x, msg.pose.position.y])

    def swarm_position_callback_1(self, msg: PoseStamped):
        self.swarm_position["drone1"] = self.convert_pose_to_grid_position(
            [msg.pose.position.x, msg.pose.position.y])

    def swarm_position_callback_2(self, msg: PoseStamped):
        self.swarm_position["drone2"] = self.convert_pose_to_grid_position(
            [msg.pose.position.x, msg.pose.position.y])

    def swarm_position_callback_3(self, msg: PoseStamped):
        self.swarm_position["drone3"] = self.convert_pose_to_grid_position(
            [msg.pose.position.x, msg.pose.position.y])

    def put_frontiers_in_grid(self):
        offsets = [
            (0, 0), (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]
        for frontier in self.frontiers:
            frontier_position = self.convert_pose_to_grid_position(frontier)
            # paint a square around the frontier

            for dx, dy in offsets:
                new_x, new_y = frontier_position[0] + dx, frontier_position[1] + dy
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:  # Ensure within bounds
                    self.grid_matrix[0, new_y, new_x] = self.FRONTIER

    def put_other_drones_in_grid(self):
        print("Drone ", self.drone_interface_list[0].drone_id,
              " Swarm position: ", self.swarm_position)
        for drone_id, position in self.swarm_position.items():
            if drone_id != self.drone_interface_list[0].drone_id:
                self.grid_matrix[0, position[1], position[0]] = self.DRONE

    def show_image_with_frontiers(self):
        image = self.process_image(self.grid_matrix)
        # centroids, frontiers = get_frontiers(image)
        # new_img = paint_frontiers(image, frontiers, centroids)
        cv2.imshow('frontiers', image)
        cv2.waitKey(10)

    def save_image_as_txt(self, path: str):
        # image = self.process_image(self.grid_matrix)
        image = image[0]
        with open(path, 'w') as f:
            for row in image:
                for cell in row:
                    f.write(f"{cell} ")
                f.write("\n")

    def save_image_as_csv(self, path: str):
        image = self.grid_matrix[0]
        np.savetxt(path, image, delimiter=",")

    # def order_and_get_frontiers_and_position(self, env_id):
    #     # Call the service to get the frontiers

    #     get_frontiers_req = GetFrontiers.Request()
    #     get_frontiers_req.explorer_id = f"drone{env_id}"
    #     get_frontiers_res = self.get_frontiers_srv.call(get_frontiers_req)
    #     self.frontiers = []
    #     self.position_frontiers = []
    #     for frontier in get_frontiers_res.frontiers:
    #         self.frontiers.append([frontier.point.x, frontier.point.y])
    #     self.frontiers = self.order_frontiers(self.frontiers)
    #     for frontier in self.frontiers:
    #         self.position_frontiers.append(self.convert_pose_to_grid_position(frontier))
    #     return self.frontiers, self.position_frontiers

    # def call_get_frontiers_with_msg(self, env_id):
    #     get_frontiers_req = GetFrontierReq()
    #     get_frontiers_req.explorer_id = f"drone{env_id}"
    #     self.get_frontiers_pub.publish(get_frontiers_req)
    #     self.wait_for_frontiers = 0
    #     self.wait_for_map = 0

    def get_frontiers_and_position_with_msg(self, env_id):
        return self.frontiers, self.position_frontiers

    def get_frontiers_and_position(self, env_id):
        get_frontiers_req = GetFrontiers.Request()
        get_frontiers_req.explorer_id = f"drone{env_id}"
        future = self.get_frontiers_srv.call_async(get_frontiers_req)
        then = self.drone_interface_list[0].get_clock().now().to_msg().sec
        while rclpy.ok():
            if future.done():
                if future.result() is not None:
                    get_frontiers_res = future.result()
                    break
                else:
                    print(f"Drone{env_id} service call failed, calling again...")
                    future = self.get_frontiers_srv.call_async(get_frontiers_req)
            if self.drone_interface_list[0].get_clock().now().to_msg().sec - then > 1.0:
                print(f"Drone{env_id} service call timeout, calling again...")
                future = self.get_frontiers_srv.call_async(get_frontiers_req)
                then = self.drone_interface_list[0].get_clock().now().to_msg().sec

        self.frontiers = []
        self.position_frontiers = []
        for frontier in get_frontiers_res.frontiers:
            self.frontiers.append([frontier.point.x, frontier.point.y])
            self.position_frontiers.append(self.convert_pose_to_grid_position([
                                           frontier.point.x, frontier.point.y]))
        return self.frontiers, self.position_frontiers

    def order_frontiers(self, frontiers):
        # Order frontiers from left to right and top to bottom
        sorted_frontiers = sorted(frontiers, key=lambda point: (-point[1], -point[0]))
        return sorted_frontiers

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
            1: [0],
            2: [128],  # Grey
            3: [50]
        }

        # Map the matrix values to the corresponding colors
        for i in range(image_matrix.shape[1]):
            for j in range(image_matrix.shape[0]):
                image[i, j] = color_map[image_matrix[i, j]]

        return image
