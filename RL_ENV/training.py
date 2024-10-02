import gymnasium as gym
from gymnasium import spaces
from torch import nn
import rclpy
import time
import numpy as np
import cProfile
import pstats
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from as2_gymnasium_env import AS2GymnasiumEnv


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


class Training:
    def __init__(self, env: AS2GymnasiumEnv, custom_callback: CustomCallback):
        self.env = env
        self.custom_callback = custom_callback

    def train(self):
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            learning_rate=0.003,
            use_sde=True,
            sde_sample_freq=8,
            policy_kwargs=dict(
                log_std_init=-1,
                ortho_init=False,
                activation_fn=th.nn.ReLU,
                net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
            )
        )
        model.learn(
            total_timesteps=30000,
            callback=self.custom_callback,
        )

        ##### PROFILING #####

        # with cProfile.Profile() as pr:

        #     model.learn(
        #         total_timesteps=1,
        #         callback=self.custom_callback,
        #     )

        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # # stats.print_stats()
        # stats.dump_stats(filename='profilings/needs_profiling_4.prof')

        ######################

        model.save("ppo_as2_gymnasium")


if __name__ == "__main__":
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world1", world_size=2.5,
                          grid_size=50, min_distance=1.0, num_envs=1, policy_type="MlpPolicy")
    env = VecMonitor(env)
    print("Start mission")
    #### ARM OFFBOARD #####
    # print("Arm")
    # env.drone_interface_list[0].offboard()
    # time.sleep(1.0)
    # print("Offboard")
    # env.drone_interface_list[0].arm()
    # time.sleep(1.0)

    # ##### TAKE OFF #####
    # print("Take Off")
    # env.drone_interface_list[0].takeoff(1.0, speed=1.0)
    # time.sleep(1.0)
    custom_callback = CustomCallback()
    training = Training(env, custom_callback)
    print("Training the model...")
    training.train()
    rclpy.shutdown()
