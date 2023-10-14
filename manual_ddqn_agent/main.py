import os
import random
import datetime
from pathlib import Path
import numpy as np
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, RemoveSeedWrapper, XValueRewardWrapper, ClipScoreboardWrapper

# https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html


# Environment setup
RESUME = True

def setup_environment():
    """Initialize and wrap the Super Mario environment."""
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
    env = ClipScoreboardWrapper(env)
    env = RemoveSeedWrapper(env)
    env = JoypadSpace(env, [['right'], ['right', 'A']])
    #env = JoypadSpace(env, RIGHT_ONLY)# - includes noop action and RIGHT + A + B
    # Moving right and jumping is enough to play the game at this point
    env = SkipFrame(env, skip=4)
    env = XValueRewardWrapper(env)  # Experimental reward shaper - simple reward based on x position scale
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)

    # Normalise observation values
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    env.reset()
    return env

def find_latest_checkpoint(dir_path):
    checkpoint_files = list(dir_path.rglob("mario_net_*.chkpt"))  # Use rglob instead of glob
    if not checkpoint_files:
        return None
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoint_files[-1]

env = setup_environment()

# Save directory setup
if RESUME:
    checkpoint_path = find_latest_checkpoint(Path('checkpoints'))
    if checkpoint_path:
        save_dir = checkpoint_path.parent
    else:
        print("No checkpoint found, starting from scratch")
        checkpoint_path = None  # Ensure it's None if no checkpoint is found
        save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        save_dir.mkdir(parents=True)
else:
    checkpoint_path = None
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

# Initialize Mario agent
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint_path)

# Initialize logger
logger = MetricLogger(save_dir, resume=RESUME)

# Training loop - run episodes manually
episodes = 40000


start_episode = 0
if checkpoint_path:
    start_episode = mario.load(checkpoint_path)
    print(f"Loaded checkpoint at episode {start_episode}")
else:
    print("No checkpoint to load from")
    print("Starting from scratch")

for e in range(start_episode, episodes):
    mario.current_episode = e
    state = env.reset()
    state = state[0]
    state = np.array(state)
    while True:
        action = mario.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.array(next_state)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q, info['x_pos'])
        state = next_state

        if done or info['flag_get']:
            break
    # metrics.py
    logger.log_episode()

    # Log every 50 episodes, but don't save until 100
    if e % 50 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
