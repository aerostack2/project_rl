import subprocess
import os

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from std_srvs.srv import SetBool, Empty

import gymnasium as gym

from typing import Any, List, Type
import math
import numpy as np
from transforms3d.euler import euler2quat
import random
import time
from copy import deepcopy

import xml.etree.ElementTree as ET

from observation import ObservationAsync as Observation
from action import DiscreteCoordinateActionSingleEnv as Action


class AS2GymnasiumEnv(gym.Env):
    def __init__(self, world_name, world_size, grid_size, min_distance, policy_type: str,
                 namespace, shared_frontiers: list = None, lock=None) -> None:
        super().__init__()
        # ROS 2 related stuff
        self.drone_interface_list = [
            DroneInterfaceTeleop(drone_id=namespace, use_sim_time=True)
        ]
        self.set_pose_client = self.drone_interface_list[0].create_client(
            SetPoseWithID, f"/world/{world_name}/set_pose"
        )
        self.world_control_client = self.drone_interface_list[0].create_client(
            ControlWorld, f"/world/{world_name}/control"
        )

        self.activate_scan_srv = self.drone_interface_list[0].create_client(
            SetBool, f"{self.drone_interface_list[0].get_namespace()}/activate_scan_to_occ_grid"
        )

        self.clear_map_srv = self.drone_interface_list[0].create_client(
            Empty, "/map_server/clear_map"
        )

        self.render_mode = "rgb_array"

        self.world_name = world_name
        self.world_size = world_size
        self.min_distance = min_distance
        self.grid_size = grid_size
        self.num_envs = 1

        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_truncated = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

        # Environment observation
        self.observation_manager = Observation(
            grid_size, 1, self.drone_interface_list, policy_type)
        self.observation_space = self.observation_manager.observation_space

        # Environment action
        self.action_manager = Action(self.drone_interface_list, self.grid_size)
        self.action_space = self.action_manager.action_space

        # Other stuff
        self.obstacles = self.parse_xml("assets/worlds/world1.sdf")
        print(self.obstacles)

        self.shared_frontiers = shared_frontiers
        self.lock = lock

    @classmethod
    def from_args(cls, args: Any) -> "AS2GymnasiumEnv":
        return cls(**vars(args))

    def pause_physics(self) -> bool:
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = True
        pause_physics_res = self.world_control_client.call(pause_physics_req)
        return pause_physics_res.success

    def unpause_physics(self) -> bool:
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = False
        pause_physics_res = self.world_control_client.call(pause_physics_req)
        return pause_physics_res.success

    def set_random_pose(self, model_name) -> tuple[bool, Pose]:
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        x = round(random.uniform(-self.world_size, self.world_size), 2)
        y = round(random.uniform(-self.world_size, self.world_size), 2)
        while True:
            too_close = any(
                self.distance((x, y), obstacle) < self.min_distance for obstacle in self.obstacles
            )
            if not too_close:
                break
            else:
                x = round(random.uniform(-self.world_size, self.world_size), 2)
                y = round(random.uniform(-self.world_size, self.world_size), 2)

        set_model_pose_req.pose.pose.position.x = x
        set_model_pose_req.pose.pose.position.y = y
        set_model_pose_req.pose.pose.position.z = 1.0
        yaw = round(random.uniform(0, 2 * math.pi), 2)
        quat = euler2quat(0, 0, yaw)
        set_model_pose_req.pose.pose.orientation.x = quat[1]
        set_model_pose_req.pose.pose.orientation.y = quat[2]
        set_model_pose_req.pose.pose.orientation.z = quat[3]
        set_model_pose_req.pose.pose.orientation.w = quat[0]

        set_model_pose_res = self.set_pose_client.call(set_model_pose_req)
        # Return success and position
        return set_model_pose_res.success, set_model_pose_req.pose.pose

    def set_pose(self, model_name, x, y) -> tuple[bool, Pose]:
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        set_model_pose_req.pose.pose.position.x = x
        set_model_pose_req.pose.pose.position.y = y
        set_model_pose_req.pose.pose.position.z = 1.0
        set_model_pose_req.pose.pose.orientation.x = 0.0
        set_model_pose_req.pose.pose.orientation.y = 0.0
        set_model_pose_req.pose.pose.orientation.z = 0.0
        set_model_pose_req.pose.pose.orientation.w = 1.0

        set_model_pose_res = self.set_pose_client.call(set_model_pose_req)
        # Return success and position
        return set_model_pose_res.success, set_model_pose_req.pose.pose

    def reset(self, seed=None, options=None):
        self.activate_scan_srv.call(SetBool.Request(data=False))
        self.pause_physics()
        self.clear_map_srv.call(Empty.Request())
        print("Resetting drone", self.drone_interface_list[0].drone_id)
        self.set_random_pose(self.drone_interface_list[0].drone_id)

        self.unpause_physics()
        self.activate_scan_srv.call(SetBool.Request(data=True))
        self.wait_for_map()
        # self.observation_manager.call_get_frontiers_with_msg(env_id=env_0)
        # while self.observation_manager.wait_for_frontiers == 0:
        #     pass
        frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
            0)
        if len(frontiers) == 0:
            return self.reset()
        obs = self._get_obs(0)
        self._save_obs(0, obs)
        return self.observation_manager._obs_from_buf(), {}

    def step(self, action):
        self.action_manager.actions = [action]
        frontier, position_frontier, path_length, result = self.action_manager.take_action(
            self.observation_manager.frontiers, self.observation_manager.position_frontiers, 0)

        with self.lock:
            self.shared_frontiers.append(position_frontier)
            self.set_pose(self.drone_interface_list[0].drone_id, frontier[0], frontier[1])

        self.wait_for_map()
        # self.observation_manager.call_get_frontiers_with_msg(env_id=0)
        # while self.observation_manager.wait_for_frontiers == 0:
        #     pass
        frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
            0)
        obs = self._get_obs(0)
        self._save_obs(0, obs)

        with self.lock:
            index = next((i for i, position_frontier in enumerate(self.shared_frontiers)
                          if self.shared_frontiers[i][0] == position_frontier[0] and self.shared_frontiers[i][1] == position_frontier[1]), None)
            print("Index", index)
            self.shared_frontiers.pop(index)
            print("Shared frontiers", self.shared_frontiers)

        self.buf_infos[0] = {}  # TODO: Add info
        self.buf_rews[0] = -path_length * 0.1
        self.buf_dones[0] = False
        self.buf_truncated[0] = False
        if len(frontiers) == 0:  # No frontiers left, episode ends
            self.buf_dones[0] = True
            self.buf_rews[0] = 100.0
            # print("entra aqui?")
            # return self.reset()
        if not result:
            print("Failed to reach goal")
            self.buf_dones[0] = True
            self.buf_rews[0] = -100.0
            # print("o entra aqui?")
            # return self.reset()
        print("Reward", self.buf_rews[0])
        return self.observation_manager._obs_from_buf(), self.buf_rews[0], self.buf_dones[0], self.buf_truncated[0], self.buf_infos[0]

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Any = None,
        **method_kwargs,
    ):
        if method_name == "action_masks":
            return self.action_masks()
        return

    def _save_obs(self, env_id, obs):
        # save the observation for the specified environment
        self.observation_manager._save_obs(obs, env_id)

    def _get_obs(self, env_id):
        # get the observation for the specified environment
        return self.observation_manager._get_obs(env_id)

    def action_masks(self):
        action_masks = np.zeros(self.action_manager.grid_size *
                                self.action_manager.grid_size, dtype=bool)
        with self.lock:
            for frontier in self.observation_manager.position_frontiers:
                if not any((frontier[0] == position_frontier[0] and frontier[1] == position_frontier[1]) for position_frontier in self.shared_frontiers):
                    action_masks[frontier[1] * self.action_manager.grid_size + frontier[0]] = True
        return action_masks

    def parse_xml(self, filename: str) -> List[tuple[float, float]]:
        """Parse XML file and return pole positions"""
        world_tree = ET.parse(filename).getroot()
        models = []
        for model in world_tree.iter('include'):
            if model.find('uri').text == 'model://pole':
                x, y, *_ = model.find('pose').text.split(' ')
                models.append((float(x), float(y)))

        return models

    def distance(self, point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """
        Calculate the euclidean distance between 2 points
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def wait_for_map(self):
        self.observation_manager.wait_for_map = 0
        while self.observation_manager.wait_for_map == 0:
            pass
        return

    def close(self):
        self.drone_interface_list[0].destroy_node()
