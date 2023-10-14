import tkinter as tk
from tkinter import filedialog
import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import matplotlib.pyplot as plt

class CustomReshapeAndResizeObs(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 80)):
        super(CustomReshapeAndResizeObs, self).__init__(env)
        old_shape = self.observation_space.shape
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, shape[0], shape[1]), dtype=np.float32)  # Adjusted channel dimension

    def observation(self, observation):
        observation = observation[32:, :]
        # Resize the observation
        observation = cv2.resize(observation, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)  # Note the order of dimensions

        # Normalize
        observation = observation.astype(np.float32) / 255.0
        # Add channel dimension
        observation = np.expand_dims(observation, axis=0)
        return observation
    
class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(FrameSkip, self).__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info

if __name__ == "__main__":
# Helper function to create environment
    def make_env(env_id, rank):
        def _init():
            env = gym.make(env_id, apply_api_compatibility=True, render_mode='human')
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            print(env.observation_space.shape)
           # env = FrameSkip(env, skip=4)
            print(env.observation_space.shape)
            env = GrayScaleObservation(env)
            print(env.observation_space.shape)
            env = CustomReshapeAndResizeObs(env, shape=(240, 256))
            print(env.observation_space.shape)
            env = RemoveSeedWrapper(env)
            env = DummyVecEnv([lambda: env])

             # Frame stacking with 4 frames
            n_stack = 4
            env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
            print(env.observation_space.shape)
            return env
        return _init

def evaluate_model(file_path, episodes=2):
    env_id = "SuperMarioBros-1-1-v0"
    env = make_env(env_id, 0)()
    
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

def show_final_observation(env_id):
    env = make_env(env_id, 0)()  # Using the make_env function you provided
    obs = env.reset()
    # Do a few random steps so that the environment is not empty
    for _ in range(500):
        action = env.action_space.sample()
        obs, _, _, _ = env.step([action])


    # If the observation is wrapped in a DummyVecEnv, take the first environment
    if len(obs.shape) == 4:
        obs = obs[0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, ax in enumerate(axes):
        ax.imshow(obs[i], cmap='gray')  # Displaying each frame
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Call the function


if __name__ == "__main__":
    show_final_observation("SuperMarioBros-1-1-v2")
    load_and_evaluate_models()
