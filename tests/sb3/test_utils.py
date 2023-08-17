import pytest
import gymnasium as gym
from unittest.mock import Mock, patch
import numpy as np

# Assuming the provided code is in a module named 'safety_evaluation'
from svf_gymnasium.sb3.utils import evaluate_safety_constrain

# Mocking the environment
@pytest.fixture
def mock_env():
    env = gym.make("InvertedPendulum-v4")
    return env


# Mocking the safety filter
@pytest.fixture
def mock_filter():
    filter = Mock()
    filter.predict.return_value = (np.array([[0]]), None)
    return filter


def test_evaluate_without_monitor(mock_env, mock_filter):
    with patch("warnings.warn") as mock_warn:
        mean_reward, std_reward = evaluate_safety_constrain(mock_filter, mock_env)
        mock_warn.assert_called_once()


def test_evaluate_with_episode_rewards(mock_env, mock_filter):
    episode_rewards, episode_lengths = evaluate_safety_constrain(
        mock_filter, mock_env, return_episode_rewards=True
    )
    assert len(episode_rewards) == 10
    assert len(episode_lengths) == 10


def test_evaluate_with_vecenv(mock_env, mock_filter):
    mock_env.num_envs = 2
    mean_reward, std_reward = evaluate_safety_constrain(mock_filter, mock_env)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)


# ... Add more tests as needed ...
