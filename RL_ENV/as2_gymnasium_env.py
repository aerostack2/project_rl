from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import gymnasium as gym
import math
from transforms3d.euler import euler2quat
from gymnasium.spaces import Box, Dict, Discrete, Tuple
from typing import Any, Iterable, List, Optional, Sequence, Type, Union
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from collections import OrderedDict
import numpy as np
import random
import time
import rclpy
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs


class AS2GymnasiumEnv(VecEnv):
    def __init__(self, world_name, world_size, grid_size, num_envs) -> None:
        self.num_envs = num_envs

        self.render_mode = []
        for _ in range(num_envs):
            self.render_mode.append("human")

        action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        observation_space = Dict(
            {
                "image": Box(
                    low=0, high=2, shape=(1, grid_size, grid_size), dtype=np.uint8
                ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown
                "position": Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                # Position of the drone in the grid
            }
        )

        self.world_size = world_size
        super().__init__(num_envs, observation_space, action_space)

        self.keys, shapes, dtypes = obs_space_info(observation_space)

        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs,) +
                 tuple(shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )

        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        # Make a drone interface with functionality to control the internal state of the drone with rl env methods
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

    def pause_physics(self):
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = True
        pause_physics_res = self.world_control_client.call(pause_physics_req)
        return pause_physics_res.success

    def unpause_physics(self):
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = False
        pause_physics_res = self.world_control_client.call(pause_physics_req)
        return pause_physics_res.success

    def set_random_pose(self, model_name):
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        set_model_pose_req.pose.pose.position.x = round(
            random.uniform(-self.world_size, self.world_size), 2
        )
        set_model_pose_req.pose.pose.position.y = round(
            random.uniform(-self.world_size, self.world_size), 2
        )
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

    def reset(self):
        pose = Pose()
        for drone in self.drone_interface_list:
            self.set_random_pose(drone.drone_id)

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
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def _save_obs(self, obs: VecEnvObs, env_id) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_id] = obs
            else:
                self.buf_obs[key][env_id] = obs[key]


if __name__ == "__main__":
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world1", world_size=3,
                          grid_size=200, num_envs=1)
    print("Start mission")

    ##### ARM OFFBOARD #####
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
        env.pause_physics()
        _, pose = env.set_random_pose("drone0")
        poseStamped = PoseStamped()
        poseStamped.pose = pose
        poseStamped.header.frame_id = "earth"
        env.drone_interface_list[
            0
        ].motion_ref_handler.position.send_position_command_with_yaw_angle(
            pose=poseStamped,
            twist_limit=0.0,
            pose_frame_id="earth",
            twist_frame_id="earth",
            yaw_angle=0.0,
        )
        print(env.buf_obs)
        env.unpause_physics()
    rclpy.shutdown()
