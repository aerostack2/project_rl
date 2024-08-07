import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose

import gymnasium as gym

from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)

from collections import OrderedDict
from typing import Any, List, Type
import math
import numpy as np
from transforms3d.euler import euler2quat
import random
import time

import xml.etree.ElementTree as ET

from observation import Observation
from action import Action


class AS2GymnasiumEnv(VecEnv):
    def __init__(self, world_name, world_size, grid_size, min_distance, num_envs) -> None:

        # ROS 2 related stuff
        self.drone_interface_list = [
            DroneInterfaceTeleop(drone_id=f"drone{n}", use_sim_time=True)
            for n in range(num_envs)
        ]
        self.set_pose_client = self.drone_interface_list[0].create_client(
            SetPoseWithID, f"/world/{world_name}/set_pose"
        )
        self.world_control_client = self.drone_interface_list[0].create_client(
            ControlWorld, f"/world/{world_name}/control"
        )

        self.render_mode = []
        for _ in range(num_envs):
            self.render_mode.append("human")

        self.world_size = world_size
        self.min_distance = min_distance

        # Environment observation
        self.observation_manager = Observation(grid_size, num_envs, self.drone_interface_list)
        observation_space = self.observation_manager.observation_space

        # Environment action
        self.action_manager = Action()
        action_space = self.action_manager.action_space

        super().__init__(num_envs, observation_space, action_space)

        self.keys = self.observation_manager.keys
        self.buf_obs = self.observation_manager.buf_obs
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        # Make a drone interface with functionality to control the internal state of the drone with rl env methods

        # Other stuff
        self.obstacles = self.parse_xml("assets/worlds/world1.sdf")
        print(self.obstacles)

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

    def reset(self) -> VecEnvObs:
        for idx, drone in enumerate(self.drone_interface_list):
            self.pause_physics()
            _, pose = self.set_random_pose(drone.drone_id)
            poseStamped = PoseStamped()
            poseStamped.pose = pose
            poseStamped.header.frame_id = "earth"
            drone.motion_ref_handler.position.send_position_command_with_yaw_angle(
                pose=poseStamped,
                twist_limit=0.0,
                pose_frame_id="earth",
                twist_frame_id="earth",
                yaw_angle=0.0,
            )
            self.unpause_physics()
            # rotate the drone to gather lidar data
            # obs = self._get_obs(idx)
            # self._save_obs(obs, idx)
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> None:
        return

    def close(self):
        return

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        try:
            return getattr(self, attr_name)
        except AttributeError:
            return None

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        return

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        return

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return

    def _obs_from_buf(self) -> VecEnvObs:
        # return all observations from all environments
        return self.observation_manager._obs_from_buf()

    def _save_obs(self, obs: VecEnvObs, env_id):
        # save the observation for the specified environment
        self.observation_manager._save_obs(obs, env_id)

    def _get_obs(self, env_id) -> VecEnvObs:
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


if __name__ == "__main__":
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world1", world_size=10,
                          grid_size=200, min_distance=1.0, num_envs=1)
    print("Start mission")
    #### ARM OFFBOARD #####
    print("Arm")
    env.drone_interface_list[0].offboard()
    time.sleep(1.0)
    print("Offboard")
    env.drone_interface_list[0].arm()
    time.sleep(1.0)

    ##### TAKE OFF #####
    print("Take Off")
    env.drone_interface_list[0].takeoff(1.0, speed=1.0)
    time.sleep(1.0)
    for i in range(10):
        env.reset()
        time.sleep(1.0)
        env.observation_manager.show_image_with_frontiers()
        time.sleep(2.0)
    rclpy.shutdown()
