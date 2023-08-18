import gymnasium as gym


class EarlyTerminationWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if cost > 0:
            terminated = True

        return obs, reward, cost, terminated, truncated, info


class BoundedRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_bound: float = 0.05):
        super().__init__(env)
        self.reward_bound = reward_bound

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info["reward"] = reward
        reward = min(self.reward_bound, reward)
        return obs, reward, cost, terminated, truncated, info
