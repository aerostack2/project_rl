from gymnasium.spaces import Box
import numpy as np


class Action:
    def __init__(self):
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
