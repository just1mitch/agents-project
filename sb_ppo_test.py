# Stable Baselines3 compatible wrapper for Super Mario Bros

import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# To get it to run needed to remove seed and options from reset, which I guess are added by DummyVecEnv
class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)

# Support for grayscale and frame stacking
class SqueezeObservation(gym.ObservationWrapper):
    def observation(self, observation):
        return np.squeeze(observation)

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = RemoveSeedWrapper(env)
env = GrayScaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = SqueezeObservation(env)

# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# For Stable Baselines3 we need to use a vectorized environment
env = DummyVecEnv([lambda: env])

state = env.reset()
# print(state.shape)
#
frame_stack = np.zeros((1, state.shape[1], state.shape[2], 10))
# fRAme stacking manually as it doesn't seem to work with the stable baselines wrapper

for step in range(100):
    action = [env.action_space.sample()]
    state, reward, done, info = env.step(action)
    frame_stack = np.roll(frame_stack, shift=-1, axis=3)
    frame_stack[:, :, :, -1] = state[:, :, :]

    if done.any():
        state = env.reset()
        frame_stack = np.zeros((1, state.shape[1], state.shape[2], 10))

# Visualize the frame stack
plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(frame_stack[0, :, :, i], cmap='gray')
    plt.title(f"Frame {i + 1}")
plt.show()
