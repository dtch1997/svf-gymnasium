import pytest
import gymnasium as gym
import svf_gymnasium.envs


@pytest.mark.parametrize(
    "env_id",
    [
        "Safe-Hopper-v4",
        "Safe-HalfCheetah-v4",
        "Safe-Walker2d-v4",
        "Safe-Ant-v4",
        "Safe-Humanoid-v4",
        "Safe-Swimmer-v4",
    ],
)
def test_env(env_id):
    env = gym.make(env_id)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()
