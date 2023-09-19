import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrapper to remove seed and options from reset
class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)

# Environment setup
test_env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
test_env = JoypadSpace(test_env, SIMPLE_MOVEMENT)
test_env = RemoveSeedWrapper(test_env)
test_env = GrayScaleObservation(test_env)
# Vectorized environment for Stable Baselines3
test_env = DummyVecEnv([lambda: test_env])

# Load the best model
best_model = PPO.load("./logs/best_model/best_model")

# Reset the environment
state = test_env.reset()

# Run the best model
for _ in range(1000):
    action, _ = best_model.predict(state)
    state, reward, done, info = test_env.step(action)
    test_env.render()
    if done.any():
        state = test_env.reset()

# Close the environment
test_env.close()
