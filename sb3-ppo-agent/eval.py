import tkinter as tk
from tkinter import filedialog
import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
import cv2

# Wrapper to remove seed and options from reset
class CustomReshapeAndResizeObs(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super(CustomReshapeAndResizeObs, self).__init__(env)
        old_shape = self.observation_space.shape
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, shape[0], shape[1]), dtype=np.float32)

    def observation(self, observation):
        # Resize the observation
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        # Normalize and reshape
        return np.expand_dims(observation.astype(np.float32) / 255.0, axis=0)
class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)

def evaluate_model(file_path, episodes=10):
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RemoveSeedWrapper(env)
    env = GrayScaleObservation(env)
    env = CustomReshapeAndResizeObs(env, shape=(128, 128))
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    model = PPO.load(file_path)
    total_distance = 0
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            env.render()
        total_distance += info[0]['x_pos']
    
    env.close()
    return total_distance / episodes

def load_and_evaluate_models():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(filetypes=[("ZIP files", "*.zip")])
    
    if not file_paths:
        print("No model files selected. Exiting.")
        return

    best_distance = 0
    best_model_path = ""
    
    for file_path in file_paths:
        avg_distance = evaluate_model(file_path)
        print(f"Model {file_path} achieved an average distance of {avg_distance:.2f}")
        
        if avg_distance > best_distance:
            best_distance = avg_distance
            best_model_path = file_path

    print(f"\nThe best model is {best_model_path} with an average distance of {best_distance:.2f}")

if __name__ == "__main__":
    load_and_evaluate_models()