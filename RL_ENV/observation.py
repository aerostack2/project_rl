import numpy as np
from gymnasium.spaces import Box, Dict
from collections import OrderedDict
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class Observation:
    def __init__(self, grid_size, num_envs, drone_interface_list):
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
        # Make the necessary subscriptions in order to get the occupancy map

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def _save_obs(self, obs: VecEnvObs, env_id):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_id] = obs
            else:
                self.buf_obs[key][env_id] = obs[key]

    # TODO: Implement the method that will return the dictionary with the observation
    def _get_obs(self, env_id) -> VecEnvObs:
        pass
