import svf_gymnasium.envs
import gymnasium as gym
import safety_gymnasium

if __name__ == "__main__":

    env = gym.make("Wrapped-SafetyPointGoal1-v0")
    print(env.unwrapped.spec)
    print("mujoco" in env.spec)
