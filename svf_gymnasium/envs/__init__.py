# Register safety envs

import gymnasium as gym
from gymnasium.envs.registration import register
from svf_gymnasium.envs import safety_wrapper
from svf_gymnasium.envs.safety_gymnasium_wrappers import *
import safety_gymnasium


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
):
    register(
        id=f"Safe-{env_id}",
        # Need to use a factory here to avoid registering the same env multiple times
        entry_point=make_safety_env_factory(env_id),
        max_episode_steps=1000,
    )


def make_wrapped_safety_gymnasium_env_factory(env_id):
    """Create a safety wrapper around a safety gymnasium environment"""

    def make_env(**kwargs):
        env = safety_gymnasium.make(env_id, **kwargs)
        env = InfoWrapper(env)
        env = BoundedRewardWrapper(env)
        env = CostPenaltyWrapper(env)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        return env

    return make_env


for task in ("Goal", "Button", "Push", "Circle"):
    for robot in ("Point", "Car", "Doggo", "Ant"):
        for level in (0, 1, 2):

            env_id = f"Safety{robot}{task}{level}-v0"
            register(
                id=f"Wrapped-{env_id}",
                # Need to use a factory here to avoid registering the same env multiple times
                entry_point=make_wrapped_safety_gymnasium_env_factory(env_id),
                max_episode_steps=1000,
            )
