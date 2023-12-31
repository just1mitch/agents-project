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
import argparse
print(torch.cuda.is_available())

import tkinter as tk
from tkinter import filedialog

# Initial PPO guide: https://github.com/nicknochnack/MarioRL/blob/main/Mario%20Tutorial.ipynb
# Heavily expanded upon and modified to suit the needs of the project

# Simple helper function to get the model path
def get_model_path():
    root = tk.Tk()
    root.withdraw()  # This hides the root window
    file_path = filedialog.askopenfilename(title="Select the Model", filetypes=[('Model Files', '*.zip')])
    return file_path

parser = argparse.ArgumentParser(description="Mario SB3-PPO Trainer")
parser.add_argument('--resume', action='store_true', help="Resume training from the selected model.")
args = parser.parse_args()

RESUME_TRAINING = args.resume
MODEL_PATH = "ppo_mario"
from gym.wrappers import ResizeObservation

# Like DDQN, but with a combined wrapper to rescale and normalize the observation
# SB3 expects an observation space of (1, 84, 84) - (channel, width, height) - 
# but at this stage the observation space doesn't have a channel due to GreyScaleObservation
class CustomReshapeAndResizeObs(gym.ObservationWrapper):
    """Reshape observation into (1, 84, 84) and rescale values into [0, 1]"""
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

# This wrapper removes the seed from the environment, which would otherwise be incompatible due to version differences
class RemoveSeedWrapper(gym.Wrapper):
    """Remove seed"""
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)

# Wrapper to enhance the reward function - successful in training without this.
# class EnhancedRewardWrapper(gym.Wrapper):
#     """Optional reward wrapper"""
#     def __init__(self, env, new_max_x_bonus=5, stuck_penalty=-3, stuck_threshold=100):
#         super(EnhancedRewardWrapper, self).__init__(env)
#         self.max_x_pos = float('-inf')  # Keep track of the maximum x position reached
#         self.last_x_pos = None  # Keep track of the last x position
#         self.stuck_counter = 0  # Counter to check if Mario is stuck
#         # Bonus for reaching a new maximum x position
#         self.new_max_x_bonus = new_max_x_bonus
#         # Penalty for staying in the same x position for too long
#         self.stuck_penalty = stuck_penalty
#         # Number of frames to consider Mario as stuck
#         self.stuck_threshold = stuck_threshold

#     def reset(self, **kwargs):
#         obs = super().reset(**kwargs)
#         self.max_x_pos = float('-inf')
#         self.last_x_pos = None
#         self.stuck_counter = 0
#         return obs

#     def step(self, action):
#         obs, reward, done, truncated, info = super().step(action)
#         x_pos = info['x_pos'] 
#         if x_pos > self.max_x_pos:
#             reward += self.new_max_x_bonus
#             self.max_x_pos = x_pos

#         # Check if Mario is stuck
#         if x_pos == self.last_x_pos:
#             self.stuck_counter += 1
#             if self.stuck_counter >= self.stuck_threshold:
#                 reward += self.stuck_penalty
#                 self.stuck_counter = 0  # Reset counter after applying penalty
#         else:
#             self.stuck_counter = 0

#         self.last_x_pos = x_pos

#         return obs, reward, done, truncated, info

if __name__ == "__main__":
# Helper function to create environment
    def make_env(env_id, rank):
        """"Make a standard env with all wrappers"""
        def _init():
            env = gym.make(env_id, apply_api_compatibility=True)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, 84)
            env = CustomReshapeAndResizeObs(env, shape=(84, 84))
          #  env = EnhancedRewardWrapper(env)
            env = RemoveSeedWrapper(env)
            print(env.observation_space.shape)
            return env
        return _init
    
# Simple Call back to save the model every X steps defined in the init: https://github.com/nicknochnack/MarioRL/blob/main/Mario%20Tutorial.ipynb
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
        
# Standard SB3 Callback for 'best model' from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
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
            


    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    # Environment setup

    # Multi Process Enviro
    env_id = 'SuperMarioBros-v0'
    # num_cpu = 2  # Number of processes to use
    env = make_env(env_id, 0)()
    # Adding Monitor wrapper to record training stats
    env = VecMonitor(DummyVecEnv([lambda: env]), LOG_DIR)
    env = VecMonitor(env, LOG_DIR)

    # Frame stacking with 4 frames, 4 seems to work best.
    # Noticed that 4 frames are stacked on the Width, not the channel. So the shape is (1, 84, 84*4) instead of (4, 84, 84)
    # However this is an acceptable input for the CNN per SB3
    n_stack = 4
    env = VecFrameStack(env, n_stack=n_stack)

    # Hyperparameters
    #PPO_3 2048, 0.00025
    #PPO_4 512, 0.00001
    #PPO_5 512, 0.000001
    #PPO_6 512, 0.0000001 w/ Reward Shape
    #...

    # Refer to /logs and /logs_old_models for TensorBoard logs of different training runs - /logs_old_models 1-10 will be the most relevant and show the different tuning attempts
    n_steps = 512 # Tested with 128, 512, 2048
    total_timesteps = 10000000000 # Set arbitrarily high number to train for as long as possible

    # entropy_coef = 0.01 - also something to tune that impacts exploration, but we have left it at the default value.

    # Initialize PPO algorithm with NatureCNN - predefined CNN architecture for an image space observation
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
        # Learning Rate suggested 0.000001 - but training was slow, final model produced with the following
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                tensorboard_log=LOG_DIR, learning_rate=0.00001, 
                n_steps=512, device="cuda")
    print("PyTorch device:", model.device)
    model.policy.to('cuda')

    #  model.lr_schedule = lambda _: new_learning_rate - is how to modify an existing models learning rate
    # Training the model

    model.learn(total_timesteps=total_timesteps, callback=[callback, callback1])

    # Save the model after training
    model.save("ppo_mario")

    # Evaluate the trained model - in reality it never gets to this point as training is stopped manually
    state = env.reset()
    for _ in range(1000):
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
       # env.render()
        if done.any():
            state = env.reset()

    env.close()
