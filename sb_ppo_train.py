import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt


class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)


class SqueezeObservation(gym.ObservationWrapper):
    def observation(self, observation):
        return np.squeeze(observation)

class ReshapeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(ReshapeObs, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        # Normalize pixel values in image
        return np.expand_dims(observation.astype(np.float32) / 255.0, axis=0)

# Environment setup
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = RemoveSeedWrapper(env)
env = GrayScaleObservation(env)
#env = ResizeObservation(env, (84, 84))

# Vectorized environment for Stable Baselines3
env = DummyVecEnv([lambda: env])

# Initialize PPO algorithm
model = PPO("MlpPolicy", env, verbose=1, device="cuda")
eval_callback = EvalCallback(env, best_model_save_path='./logs/best_model',
                             log_path='./logs/results', eval_freq=10000)

model.learn(total_timesteps=200000, callback=eval_callback)
model.save("ppo_mario")
state = env.reset()
for _ in range(1000):
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
    if done.any():
        state = env.reset()
env.close()
