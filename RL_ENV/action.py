from gymnasium.spaces import Box
import numpy as np
from as2_msgs.action import NavigateToPoint
from rclpy.action import ActionClient
import random


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


class Action:
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

        return frontier, path_length

    def generate_random_action(self):
        return [np.array([random.uniform(0, 1)])]

    def select_frontier(self, frontier_list, action):
        index = int(round(action[0] * (len(frontier_list) - 1)))

        return frontier_list[index]
