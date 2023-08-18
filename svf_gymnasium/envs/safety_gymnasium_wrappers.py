import gymnasium as gym

UNSAFE_PENALTY = (0.05 - (-0.05)) / (1 - 0.99)


class InfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info["reward"] = reward
        # info['cost'] is added by SafetyGymnasium2Gymnasium wrapper
        return obs, reward, cost, terminated, truncated, info


class EarlyTerminationWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if cost > 0:
            terminated = True

        return obs, reward, cost, terminated, truncated, info


class CostPenaltyWrapper(gym.Wrapper):
    def __init__(
        self, env, cost_limit: float = 1, cost_penalty: float = UNSAFE_PENALTY
    ):
        super().__init__(env)
        self.cost_limit = cost_limit
        self.cost_penalty = cost_penalty

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if cost > self.cost_limit:

            reward -= self.cost_penalty
        return obs, reward, cost, terminated, truncated, info


class BoundedRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_bound: float = 0.05):
        super().__init__(env)
        self.reward_bound = reward_bound

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        reward = min(self.reward_bound, reward)
        reward = max(-self.reward_bound, reward)
        return obs, reward, cost, terminated, truncated, info
