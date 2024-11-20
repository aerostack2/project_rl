import functools
from transforms3d.euler import euler2quat
import random
import math
from copy import deepcopy
import xml.etree.ElementTree as ET
from typing import List

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from action import DiscreteCoordinateActionSingleEnv
from observation import Observation

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_srvs.srv import SetBool, Empty


class AS2AECENV(AECEnv):
    metadata = {"render_modes": ["human"], "name": "AS2AECENV"}

    def __init__(self, render_mode=None):

        self.render_mode = render_mode

        self.possible_agents = ["agent_" + str(r) for r in range(4)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.drone_interface_list = [
            DroneInterfaceTeleop(drone_id=f"drone{n}", use_sim_time=True)
            for n in range(4)
        ]

        self.activate_scan_srv_list = [
            self.drone_interface_list[n].create_client(
                SetBool, f"{self.drone_interface_list[n].get_namespace()}/activate_scan_to_occ_grid"
            )
            for n in range(4)
        ]

        self.set_pose_client = self.drone_interface_list[0].create_client(
            SetPoseWithID, f"/world/world2/set_pose"
        )
        self.world_control_client = self.drone_interface_list[0].create_client(
            ControlWorld, f"/world/world2/control"
        )

        self.clear_map_srv = self.drone_interface_list[0].create_client(
            Empty, "/map_server/clear_map"
        )

        self.grid_size = 200

        self.action_manager = DiscreteCoordinateActionSingleEnv(
            self.drone_interface_list, self.grid_size)

        self._action_spaces = {
            agent: self.action_manager.action_space for agent in self.possible_agents
        }

        self.observation_manager = Observation(
            self.grid_size, 4, self.drone_interface_list, "MultiInputPolicy")

        self._observation_spaces = {
            agent: self.observation_manager.observation_space for agent in self.possible_agents
        }
        world_name = "world2"
        world_size = 10.0
        min_distance = 1.0
        grid_size = 200

        self.world_name = world_name
        self.world_size = world_size
        self.min_distance = min_distance
        self.grid_size = grid_size

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

    def set_random_pose(self, model_name, drones: list[tuple[float, float]]) -> tuple[bool, Point]:
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        x = round(random.uniform(-self.world_size, self.world_size), 2)
        y = round(random.uniform(-self.world_size, self.world_size), 2)
        while True:
            too_close = any(
                self.distance((x, y), obstacle) < self.min_distance for obstacle in self.obstacles + drones
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
        return set_model_pose_res.success, set_model_pose_req.pose.pose.position

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def reset(self):

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        for idx, _ in enumerate(self.drone_interface_list):
            self.activate_scan_srv_list[idx].call(SetBool.Request(data=False))

        self.pause_physics()
        self.clear_map_srv.call(Empty.Request())

        drones = []
        for idx, _ in enumerate(self.drone_interface_list):
            _, position = self.set_random_pose(self.drone_interface_list[idx].drone_id, drones)
            drones.append((position.x, position.y))

        self.unpause_physics()

        for idx, _ in enumerate(self.drone_interface_list):
            self.activate_scan_srv_list[idx].call(SetBool.Request(data=True))

        self.wait_for_map()
        frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
            0)

        if len(frontiers) == 0:
            return self.reset()

        for idx, _ in enumerate(self.drone_interface_list):
            obs = self._get_obs(idx)
            self.observations[self.agents[idx]] = obs
            self._save_obs(idx, obs)

        self._agent_selector = agent_selector(self.agents)
        self._agent_selector.reset()

    def step(self, action):
        idx = self.agents.index(agent)
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        self.action_manager.actions.append(action)
        frontier, path_length, result = self.action_manager.take_action(
            self.observation_manager.frontiers, self.observation_manager.position_frontiers, idx)

        self.set_pose(self.drone_interface_list[idx].drone_id, frontier[0], frontier[1])
        self.wait_for_map()
        # self.observation_manager.call_get_frontiers_with_msg(env_id=idx)
        # while self.observation_manager.wait_for_frontiers == 0:
        #     pass
        frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
            idx)
        obs = self._get_obs(idx)
        self._save_obs(idx, obs)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def _obs_from_buf(self):
        # return all observations from all environments
        return self.observation_manager._obs_from_buf()

    def _save_obs(self, env_id, obs):
        # save the observation for the specified environment
        self.observation_manager._save_obs(obs, env_id)

    def _get_obs(self, env_id):
        # get the observation for the specified environment
        return self.observation_manager._get_obs(env_id)

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
