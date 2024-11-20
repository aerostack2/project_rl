import gymnasium as gym
from gymnasium import spaces
from torch import nn
import rclpy
import time
import numpy as np
import cProfile
import pstats
import torch as th

# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from sb3_contrib.common.envs import InvalidActionEnvDiscrete

from as2_gymnasium_env_discrete import AS2GymnasiumEnv

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        # Access the actual environment from the callback's training environment
        pass

    def _on_step(self) -> bool:
        # Get the done flags and rewards for all environments
        pass

        return True

    def _on_rollout_end(self) -> None:
        # Compute and log the means of the rewards and lengths
        pass

class Test:
    def __init__(self, env: AS2GymnasiumEnv, custom_callback: CustomCallback, path: str):
        self.env = env
        self.custom_callback = custom_callback
        self.model = MaskablePPO.load(path, self.env)

    def test(self):
        mean_reward, std_reward = evaluate_policy(self.model, self.env, 100)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world1", world_size=2.5,
                          grid_size=50, min_distance=1.0, num_envs=1, policy_type="MultiInputPolicy")
    env = VecMonitor(env)
    custom_callback = CustomCallback()
    test = Test(env, custom_callback, "tensorboard/TRAINING_9_DISCRETE/ppo_as2_gymnasium.zip")
    test.test()
    rclpy.shutdown()