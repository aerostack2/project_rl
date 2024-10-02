from gymnasium.spaces import Box
import numpy as np
from as2_msgs.action import NavigateToPoint
from rclpy.action import ActionClient
import random
import math
from geometry_msgs.msg import PointStamped


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
        return result.result.success, result.result.path_length.data


class ActionSingleValue:
    def __init__(self, drone_interface_list):
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.drone_interface_list = drone_interface_list
        self.actions = None
        self.generate_path_action_client_list = []
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )

    def take_action(self, frontier_list, env_id) -> tuple:
        action = self.actions[env_id]
        frontier = self.select_frontier(frontier_list, action)

        result, path_length = self.generate_path_action_client_list[env_id].send_goal(frontier)

        return frontier, path_length, result

    def generate_random_action(self):
        return [np.array([random.uniform(0, 1)])]

    def select_frontier(self, frontier_list, action):
        index = int(round(action[0] * (len(frontier_list) - 1)))

        return frontier_list[index]


class ActionScalarVector:
    def __init__(self, drone_interface_list):
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.drone_interface_list = drone_interface_list
        self.actions = None
        self.generate_path_action_client_list = []
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def take_action(self, frontiers, world_size, env_id):
        action = self.actions[env_id]

        angle_action = math.atan2(action[1], action[0])
        magnitude_action = math.sqrt(action[0] ** 2 + action[1] ** 2)

        # for frontier in frontier_positions:
        #     frontier[0] = frontier[0] - grid_size / 2
        #     frontier[1] = -(frontier[1] - grid_size / 2)

        frontier_index, closest_distance = self.select_frontier(
            angle_action, magnitude_action, frontiers, world_size)

        result, path_length = self.generate_path_action_client_list[env_id].send_goal(
            frontiers[frontier_index])

        return frontiers[frontier_index], path_length, closest_distance, result

    def select_frontier(self, angle_action, magnitude_action, frontiers, world_size):
        closest_index = None
        closest_distance = float("inf")
        x_action = magnitude_action * math.cos(angle_action)
        y_action = magnitude_action * math.sin(angle_action)

        for index, frontier in enumerate(frontiers):
            angle_frontier = math.atan2(
                frontier[1] / world_size, frontier[0] / world_size)
            magnitude_frontier = math.sqrt(
                (frontier[0] / world_size) ** 2 + (frontier[1] / world_size) ** 2)

            x_frontier = magnitude_frontier * math.cos(angle_frontier)
            y_frontier = magnitude_frontier * math.sin(angle_frontier)

            point_distance = math.sqrt((x_frontier - x_action) ** 2 + (y_frontier - y_action) ** 2)

            if point_distance < closest_distance:
                closest_distance = point_distance
                closest_index = index

        chosen_action = PointStamped()
        chosen_action.header.frame_id = "earth"
        chosen_action.point.x = x_action * world_size
        chosen_action.point.y = y_action * world_size
        self.chosen_action_pub.publish(chosen_action)

        return closest_index, closest_distance
