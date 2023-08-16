# Register safety envs

import gymnasium as gym
from gymnasium.envs.registration import register
from svf_gymnasium.envs import safety_wrapper


def make_safety_env(env_id):
    """Create a safety wrapper around an environment"""
    return safety_wrapper.BoundedRewardWrapper(gym.make(env_id))


for env_id in (
    "Hopper-v4",
    "HalfCheetah-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
):
    register(
        id=f"Safe-{env_id}",
        entry_point=lambda: make_safety_env(env_id),
        max_episode_steps=1000,
    )
