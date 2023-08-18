import safety_gymnasium
import gymnasium as gym
import gymnasium_robotics
import time

from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium

if __name__ == "__main__":

    env = safety_gymnasium.make("SafetyDoggoGoal0-v0", render_mode="rgb_array")
    print(env.metadata)
    env = SafetyGymnasium2Gymnasium(env)
    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: True)
    """
    Vision Environment
        env = safety_gymnasium.make('SafetyPointCircle0Vision-v0', render_mode='human')
    Keyboard Debug environment
    due to the complexity of the agent's inherent dynamics, only partial support for the agent.
        env = safety_gymnasium.make('SafetyPointCircle0Debug-v0', render_mode='human')
    """
    obs, info = env.reset()
    # Set seeds
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    for _ in range(1000):
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # modified for Safe RL, added cost
        obs, reward, terminated, truncated, info = env.step(act)
        env.render()
        ep_ret += reward
        ep_cost += info["cost"]
        if terminated or truncated:
            print(f"Episode return: {ep_ret}, Episode cost: {ep_cost}")
            ep_ret, ep_cost = 0, 0
            observation, info = env.reset()

    env.close()
