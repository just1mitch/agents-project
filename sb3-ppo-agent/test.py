import tkinter as tk
from tkinter import filedialog
import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
import cv2

# Basic Test Script to load and run a model.

# This was used initially to test the model and to see if it was working as intended, it was later replaced by eval.py which is a more advanced version of this script

# Wrapper to remove seed and options from reset

# Sake of simplicity, we use the same wrappers as main.py
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

def load_and_run_model():
    # Open a file dialog and get the selected file path
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
    
    if not file_path:
        print("No model file selected. Exiting.")
        return

    # Environment setup
    test_env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    test_env = JoypadSpace(test_env, SIMPLE_MOVEMENT)
    test_env = RemoveSeedWrapper(test_env)
    #test_env = MarioRewardShapingWrapper(test_env)
    test_env = GrayScaleObservation(test_env)
    #Resize to 128
    test_env = ResizeObservation(test_env, shape=(128, 128))
    test_env = CustomReshapeAndResizeObs(test_env, shape=(128, 128))
    # Vectorized environment for Stable Baselines3
    test_env = DummyVecEnv([lambda: test_env])
    n_stack = 4
    test_env = VecFrameStack(test_env, n_stack=n_stack)

    # Load the selected model
    best_model = PPO.load(file_path)

    # Reset the environment
    state = test_env.reset()

# Run the selected model
    for _ in range(100000):
        action, _ = best_model.predict(state, deterministic=False)
        action_to_take = int(action[0]) if np.ndim(action) > 0 else int(action)
        state, reward, done, info = test_env.step([action_to_take])
        print(info)
        print(reward)
        test_env.render()
        if done.any():
            state = test_env.reset()



    # Close the environment
    test_env.close()

if __name__ == "__main__":
    load_and_run_model()
