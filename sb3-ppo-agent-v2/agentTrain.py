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

RESUME_TRAINING = False
MODEL_PATH = "ppo_mario"
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
            env = gym.make(env_id, apply_api_compatibility=True)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
        #   env = EnhancedRewardWrapper(env)
            print(env.observation_space.shape)
           # env = FrameSkip(env, skip=4)
            print(env.observation_space.shape)
            env = GrayScaleObservation(env)
            print(env.observation_space.shape)
            env = CustomReshapeAndResizeObs(env, shape=(240, 256))
            print(env.observation_space.shape)
            env = RemoveSeedWrapper(env)
            env = VecMonitor(DummyVecEnv([lambda: env]), CHECKPOINT_DIR)
             # Frame stacking with 4 frames
            n_stack = 4
            env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
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
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        :param check_freq:
        :param log_dir: Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, "best_model")
            self.best_mean_reward = -np.inf

        def _init_callback(self) -> None:
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    if self.verbose >= 1:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose >= 1:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)

                return True
            


    CHECKPOINT_DIR = './checkpoints/'
    LOG_DIR = './logs/'
    # Environment setup


    env_id = 'SuperMarioBros-1-1-v2'
    env = make_env(env_id, 0)()


    # Hyperparameters
    #PPO_3 2048, 0.00025
    n_steps = 512
    learning_rate = 0.0001
    ent_coef = 0.01
    vf_coef = 0.5
    total_timesteps = 10000000

    # Initialize PPO algorithm with NatureCNN
    policy_kwargs = {
        "normalize_images": False
    }
    callback = TrainAndLoggingCallback(check_freq=25000, save_path=CHECKPOINT_DIR, verbose=1)
    callback1 = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=CHECKPOINT_DIR)
    # Create the model

    if RESUME_TRAINING:
        getModel = get_model_path()
        if os.path.exists(getModel):
            print("Loading existing model." + getModel)
            model = PPO.load(getModel, env=env, tensorboard_log=LOG_DIR)
            #Set the learning rate
            model.lr_schedule = lambda _: 0.000001
            model.ent_coef = 0.01
            
        else:
            # If the model doesn't exist, create a new one
            print("Model not found. Creating a new one.")
    else:
        print("Creating a new model.")
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                tensorboard_log=LOG_DIR, learning_rate=learning_rate, ent_coef=0.01, 
                n_steps=512, device="cuda")
    print("PyTorch device:", model.device)
    model.policy.to('cuda')

    #   new_n_steps = 512
    #  model.lr_schedule = lambda _: new_learning_rate
    # Training the model
    model.learn(total_timesteps=total_timesteps, callback=[callback, callback1])

    # Save the model after training
    model.save("ppo_mario")

    # # Evaluate the trained model
    # state = env.reset()
    # for _ in range(1000):
    #     action, _ = model.predict(state)
    #     state, reward, done, truncated, info = env.step(action)
    #    # env.render()
    #     if done.any():
    #         state = env.reset()

    # env.close()
