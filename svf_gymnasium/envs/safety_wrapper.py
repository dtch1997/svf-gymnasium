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

    def __init__(self, env: Env, strategy: str = "clip", scale=5):
        super().__init__(env)
        self.strategy = strategy
        self.reward_transform = reward_transforms[strategy]
        self.scale = scale

    def reward(self, reward):
        return self.reward_transform(reward / self.scale)


class TerminationPenaltyWrapper(gym.Wrapper):
    """Wrapper that penalizes termination"""

    def __init__(self, env: Env, penalty: float = -1):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward += self.penalty
        return observation, reward, terminated, info
