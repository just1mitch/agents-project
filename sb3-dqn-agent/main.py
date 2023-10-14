import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, FrameStack
from gym_super_mario_bros import make

# Set up the environment
def make_mario_env():
    env = make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)  # Stack 4 frames
    return env

# Create main and evaluation environments
env = DummyVecEnv([make_mario_env])
env = VecFrameStack(env, n_stack=4)

eval_env = DummyVecEnv([make_mario_env])
eval_env = VecFrameStack(eval_env, n_stack=4)

# Define the DQN model
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    verbose=1,
    tensorboard_log="./mario_tensorboard/",
)

# Create callback for evaluation during training
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=5000,  # Evaluate the model every 5000 steps
    n_eval_episodes=5,
    deterministic=True,
)

# Train the model
model.learn(total_timesteps=int(1e6), callback=eval_callback)

# Save the model after training
model.save("mario_dqn_model")

print("Training complete!")

# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()
