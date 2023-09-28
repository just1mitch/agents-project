import gym
import numpy as np
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import NatureCNN
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from gym.wrappers import GrayScaleObservation
import torch
import cv2
from stable_baselines3.common.evaluation import evaluate_policy
print(torch.cuda.is_available())

import tkinter as tk
from tkinter import filedialog
def get_model_path():
    root = tk.Tk()
    root.withdraw()  # This hides the root window
    file_path = filedialog.askopenfilename(title="Select the Model", filetypes=[('Model Files', '*.zip')])
    return file_path

RESUME_TRAINING = True
MODEL_PATH = "ppo_mario"
from gym.wrappers import ResizeObservation

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

class EnhancedRewardWrapper(gym.Wrapper):
    def __init__(self, env, new_max_x_bonus=5, stuck_penalty=-3, stuck_threshold=100):
        super(EnhancedRewardWrapper, self).__init__(env)
        self.max_x_pos = float('-inf')  # Keep track of the maximum x position reached
        self.last_x_pos = None  # Keep track of the last x position
        self.stuck_counter = 0  # Counter to check if Mario is stuck
        # Bonus for reaching a new maximum x position
        self.new_max_x_bonus = new_max_x_bonus
        # Penalty for staying in the same x position for too long
        self.stuck_penalty = stuck_penalty
        # Number of frames to consider Mario as stuck
        self.stuck_threshold = stuck_threshold

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.max_x_pos = float('-inf')
        self.last_x_pos = None
        self.stuck_counter = 0
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        x_pos = info['x_pos'] 

        # Bonus for new max x position
        if x_pos > self.max_x_pos:
            reward += self.new_max_x_bonus
            self.max_x_pos = x_pos

        # Check if Mario is stuck
        if x_pos == self.last_x_pos:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                reward += self.stuck_penalty
                self.stuck_counter = 0  # Reset counter after applying penalty
        else:
            self.stuck_counter = 0  # Reset counter if Mario moved

        self.last_x_pos = x_pos  # Update the last x position for the next step

        return obs, reward, done, truncated, info

if __name__ == "__main__":
# Helper function to create environment
    def make_env(env_id, rank):
        def _init():
            env = gym.make(env_id, apply_api_compatibility=True)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, 128)
            env = CustomReshapeAndResizeObs(env, shape=(128, 128))
            env = EnhancedRewardWrapper(env)
            env = RemoveSeedWrapper(env)
            print(env.observation_space.shape)
            return env
        return _init
    
    class TrainAndLoggingCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(TrainAndLoggingCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path
            
        def _init_callback(self):
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
        
        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
                self.model.save(model_path)
            
            return True

    class SaveOnBestTrainingRewardCallback(BaseCallback):
        def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, 'best_model')
            self.best_mean_reward = -np.inf

        def _init_callback(self):
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)
            return True   
            


    CHECKPOINT_DIR = './train17/'
    LOG_DIR = './logs/'
    # Environment setup


    env_id = 'SuperMarioBros-v0'
    num_cpu = 2  # Number of processes to use
    env = make_env(env_id, 0)()
    env = VecMonitor(DummyVecEnv([lambda: env]), LOG_DIR)
    env = VecMonitor(env, LOG_DIR)

    # Frame stacking with 4 frames
    n_stack = 4
    env = VecFrameStack(env, n_stack=n_stack)

    # Hyperparameters
    #PPO_3 2048, 0.00025
    n_steps = 512
    total_timesteps = 10000000000

    # Initialize PPO algorithm with NatureCNN
    policy_kwargs = {
        "features_extractor_class": NatureCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": False
    }
    # eval_env = make_env('SuperMarioBros-v0', 99)()  # Create a separate environment for evaluation
    # eval_env = DummyVecEnv([lambda: eval_env])
    # n_stack = 4
    # eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    # Initialize callbacks and directories

    callback = TrainAndLoggingCallback(check_freq=25_000, save_path=CHECKPOINT_DIR, verbose=1)
    callback1 = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=LOG_DIR)
    # Create the model

    if RESUME_TRAINING:
        getModel = get_model_path()
        if os.path.exists(getModel):
            print("Loading existing model." + getModel)
            model = PPO.load(getModel, env=env, tensorboard_log=LOG_DIR)
        else:
            # If the model doesn't exist, create a new one
            print("Model not found. Creating a new one.")
    else:
        print("Creating a new model.")
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                tensorboard_log=LOG_DIR, learning_rate=0.00001, 
                n_steps=512, device="cuda")
    print("PyTorch device:", model.device)
    model.policy.to('cuda')

  #   new_n_steps = 512
  #  model.lr_schedule = lambda _: new_learning_rate
    # Training the model
    model.learn(total_timesteps=total_timesteps, callback=[callback, callback1])

    # Save the model after training
    model.save("ppo_mario")

    # Evaluate the trained model
    state = env.reset()
    for _ in range(1000):
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
       # env.render()
        if done.any():
            state = env.reset()

    env.close()
