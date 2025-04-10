from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
from as2_msgs.action import NavigateToPoint
from rclpy.action import ActionClient
import random
import math
from geometry_msgs.msg import PointStamped, Point
# from rdp import rdp
import time


class PathActionClient:
    def __init__(self, drone_interface):
        self._action_client = ActionClient(
            drone_interface,
            NavigateToPoint,
            f"{drone_interface.get_namespace()}/navigate_to_point",
        )
        self.drone_interface = drone_interface

    def send_goal(self, point: list[float, float]):
        goal_msg = NavigateToPoint.Goal()
        goal_msg.point.point.x = point[0]
        goal_msg.point.point.y = point[1]
        goal_msg.point.header.frame_id = "earth"
        self._action_client.wait_for_server()
        result = self._action_client.send_goal(goal_msg)
        return result.result.success, result.result.path_length.data, result.result.path


class NearestFrontierAction:
    def __init__(self, drone_interface_list, grid_size):
        self.dims = [grid_size, grid_size]
        self.action_space = Discrete(grid_size * grid_size)
        self.drone_interface_list = drone_interface_list
        self.actions = []
        self.generate_path_action_client_list = []
        self.grid_size = grid_size
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def convert_grid_position_to_pose(self, grid_position: np.ndarray) -> list[float]:
        desp = self.grid_size / 2
        # grid_position[0] corresponds to x (derived from pose[1])
        # grid_position[1] corresponds to y (derived from pose[0])
        pose_1 = (desp - grid_position[0]) / 10.0  # This recovers pose[1]
        pose_0 = (desp - grid_position[1]) / 10.0  # This recovers pose[0]
        return [pose_0, pose_1]

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id) -> tuple:
        position = self.drone_interface_list[0].position
        # Get closest frontier position based on euclidean distance
        closest_distance = np.inf
        for i in range(len(frontier_list)):
            grid_frontier = frontier_list[i]
            grid_frontier = np.array(grid_frontier)
            distance = np.linalg.norm(np.array([position[0], position[1]]) - grid_frontier)
            if distance < closest_distance:
                closest_distance = distance
                action = i
        # Find the index of the closest frontier

        frontier = frontier_list[action]
        # result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(frontier)
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)
        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            # path_simplified = rdp(nav_path, epsilon=0.1)
            path_length = self.path_length(nav_path)

        return frontier, path_length, result

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))


class HybridAction:
    def __init__(self, drone_interface_list, grid_size):
        self.dims = [grid_size, grid_size]
        self.action_space = Discrete(grid_size * grid_size)
        self.drone_interface_list = drone_interface_list
        self.actions = []
        self.generate_path_action_client_list = []
        self.grid_size = grid_size
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def compute_information_gain(self, frontier, occupancy_grid, sensor_range):
        ig = 0
        for dx in range(-sensor_range, sensor_range + 1):
            for dy in range(-sensor_range, sensor_range + 1):
                x, y = frontier[0] + dx, frontier[1] + dy
                if 0 <= x < occupancy_grid.shape[0] and 0 <= y < occupancy_grid.shape[1]:
                    if occupancy_grid[x, y] == -1:  # unknown cell
                        ig += 1
        return ig

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id, occupancy_grid, sensor_range=5, lambda_weight=0.5) -> tuple:
        position = self.drone_interface_list[0].position
        best_utility = -np.inf
        best_frontier_idx = None
        best_info_gain = 0
        best_distance = 0

        for i, grid_frontier in enumerate(frontier_list):
            grid_frontier = np.array(grid_frontier)
            distance = np.linalg.norm(np.array([position[0], position[1]]) - grid_frontier)
            info_gain = self.compute_information_gain(grid_frontier, occupancy_grid, sensor_range)

            utility = info_gain * np.exp(-lambda_weight * distance)

            if utility > best_utility:
                best_utility = utility
                best_frontier_idx = i
                best_info_gain = info_gain
                best_distance = distance

        frontier = frontier_list[best_frontier_idx]
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)

        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            path_length = self.path_length(nav_path)

        return frontier, path_length, result

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
