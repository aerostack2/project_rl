import rclpy
import cProfile
import pstats
import torch as th
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from as2_gymnasium_env_discrete import AS2GymnasiumEnv
from custom_cnn import CustomCombinedExtractor

import argparse


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

    # def mask_fn(env: AS2GymnasiumEnv) -> np.ndarray:
    #     # Do whatever you'd like in this function to return the action mask
    #     # for the current env. In this example, we assume the env has a
    #     # helpful method we can rely on.
    #     return env.valid_action_mask()

    def train(self, n_steps: int = 128, batch_size: int = 32, n_epochs: int = 5, learning_rate: float = 0.00005, pi_net_arch: list = [256, 128, 128], vf_net_arch: list = [256, 128, 128]):
        print(
            f"Training with n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, learning_rate={learning_rate}, pi_net_arch={pi_net_arch}, vf_net_arch={vf_net_arch}")
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            self.env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            policy_kwargs=dict(
                activation_fn=th.nn.ReLU,
                net_arch=dict(pi=pi_net_arch, vf=vf_net_arch),
                features_extractor_class=CustomCombinedExtractor
            )
        )
        model.learn(
            total_timesteps=30000,
            callback=self.custom_callback,
        )

        ##### PROFILING #####

        # with cProfile.Profile() as pr:

        #     model.learn(
        #         total_timesteps=10,
        #         callback=self.custom_callback,
        #     )

        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # # stats.print_stats()
        # stats.dump_stats(
        #     filename='profilings/needs_profiling_no_clock_sim_time.prof')

        ######################

        model.save("ppo_as2_gymnasium")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform training of the model")
    parser.add_argument("--n_steps", type=int, default=128,
                        help="Number of steps in the environment")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--pi_net_arch", type=list,
                        default=[128, 128], help="Policy network architecture")
    parser.add_argument("--vf_net_arch", type=list,
                        default=[128, 128], help="Value function network architecture")
    args = parser.parse_args()

    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world1", world_size=2.5,
                          grid_size=50, min_distance=1.0, num_envs=1, policy_type="MultiInputPolicy")
    #
    env = VecMonitor(env)
    # env = ActionMasker(env.venv, action_mask_fn=Training.mask_fn)
    print("Start mission")
    custom_callback = CustomCallback()
    training = Training(env, custom_callback)
    print("Training the model...")
    training.train(n_steps=args.n_steps, batch_size=args.batch_size,
                   n_epochs=args.n_epochs, learning_rate=args.learning_rate,
                   pi_net_arch=args.pi_net_arch, vf_net_arch=args.vf_net_arch)
    rclpy.shutdown()
