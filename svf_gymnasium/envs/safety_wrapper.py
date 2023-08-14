from gymnasium.core import Env
import numpy as np
import gymnasium as gym


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip(x):
    return np.clip(x, 0, 1)


reward_transforms = {
    "sigmoid": sigmoid,
    "clip": clip,
}


class BoundedRewardWrapper(gym.RewardWrapper):
    """Wrapper that squashes rewards to [0, 1]"""

    def __init__(self, env: Env, strategy: str = "sigmoid"):
        super().__init__(env)
        self.strategy = strategy
        self.reward_transform = reward_transforms[strategy]

    def reward(self, reward):
        return self.reward_transform(reward)
