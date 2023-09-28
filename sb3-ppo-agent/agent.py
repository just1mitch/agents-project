import gym
import numpy as np
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import NatureCNN
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import cv2
print(torch.cuda.is_available())
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


if __name__ == "__main__":
# Helper function to create environment
    def make_env(env_id, rank):
        def _init():
            env = gym.make(env_id, apply_api_compatibility=True)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = GrayScaleObservation(env)
            env = CustomReshapeAndResizeObs(env, shape=(84, 84))
            env = RemoveSeedWrapper(env)
            print(env.observation_space.shape)
            return env
        return _init

    class TrainAndLoggingCallback(BaseCallback):
        def __init__(self, check_freq, save_path, eval_env, n_eval_episodes=10, verbose=1):
            super(TrainAndLoggingCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path
            self.best_mean_reward = -float("inf")  # Keep track of the best mean reward
            self.eval_env = eval_env  # Evaluation environment
            self.n_eval_episodes = n_eval_episodes  # Number of episodes to evaluate

        def _init_callback(self):
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                # Save the model at regular intervals
                regular_save_path = os.path.join(self.save_path, 'model_{}'.format(self.n_calls))
                self.model.save(regular_save_path)

                # Evaluate the model
                mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)

                # If the mean reward is higher than the previous best, save the model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.save_path, 'best_model')
                    self.model.save(best_model_path)
                    print(f"New best model! Mean reward: {mean_reward:.2f}, Total timesteps: {self.n_calls}")

            return True

    
    # Environment setup
    env_id = 'SuperMarioBros-v0'
    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Frame stacking with 4 frames
    n_stack = 4
    env = VecFrameStack(env, n_stack=n_stack)

    eval_env = make_env('SuperMarioBros-v0', 99)()  # Create a separate environment for evaluation
    eval_env = DummyVecEnv([lambda: eval_env])
    n_stack = 4
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    print(eval_env.observation_space.shape)

    # Hyperparameters
    learning_rate = 0.000001
    #PPO_3 2048, 0.00025
    n_steps = 1024
    total_timesteps = 100000000
    # Initialize PPO algorithm with NatureCNN
    policy_kwargs = {
        "features_extractor_class": NatureCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": False
    }

    # Initialize callbacks and directories
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR, eval_env=eval_env, verbose=1)
    # Create the model
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                tensorboard_log=LOG_DIR, learning_rate=0.000001, 
                n_steps=512, device="cuda")
    print("PyTorch device:", model.device)
    model.policy.to('cuda')

    # Training the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

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
