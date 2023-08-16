# Register safety envs

import gymnasium as gym
from gymnasium.envs.registration import register
from svf_gymnasium.envs import safety_wrapper


def make_safety_env_factory(env_id):
    """Create a safety wrapper around an environment"""
    return lambda **kwargs: safety_wrapper.BoundedRewardWrapper(
        gym.make(env_id, **kwargs)
    )


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
        # Need to use a factory here to avoid registering the same env multiple times
        entry_point=make_safety_env_factory(env_id),
        max_episode_steps=1000,
    )
