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
import argparse

# Much of the agent is derived from the following sources:
# https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# https://github.com/yfeng997/MadMario/blob/master/agent.py
# https://www.statworx.com/en/content-hub/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/

# Check the resume flag
parser = argparse.ArgumentParser(description="Mario DDQN Trainer")
parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint")
args = parser.parse_args()

RESUME = args.resume

# Set up the environment with the wrappers
def setup_environment():
    """Initialize and wrap the Super Mario environment."""
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
    env = ClipScoreboardWrapper(env)
    env = RemoveSeedWrapper(env)
    env = JoypadSpace(env, [['right'], ['right', 'A']])
    #env = JoypadSpace(env, RIGHT_ONLY)# - includes noop action and RIGHT + A + B
    # Moving right and jumping is enough to play the game at this point, but it could be good to test the other actionspace
    env = SkipFrame(env, skip=4) # Skip 4 frames at a time to speed up training
    env = XValueRewardWrapper(env)  # Experimental reward shaper - simple reward based on x position scale
    env = GrayScaleObservation(env, keep_dim=False) # Convert to grayscale, removes colour channel
    env = ResizeObservation(env, shape=84) # Resize to 84x84 (technically, 84x80 is the space, but 4 pixel rescale is negligible)

    # Normalise observation values
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    env.reset()
    return env

def find_latest_checkpoint(dir_path):
    checkpoint_files = list(dir_path.rglob("mario_net_*.chkpt"))  # Use rglob instead of glob just in case we have subdirectories
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


# Adjust the number of episodes to run here
episodes = 40000


start_episode = 0
# If we're resuming, load the latest checkpoint:
if checkpoint_path:
    start_episode = mario.load(checkpoint_path)
    print(f"Loaded checkpoint at episode {start_episode}")
else:
    print("No checkpoint to load from")
    print("Starting from scratch")

# Training loop
for e in range(start_episode, episodes):
    mario.current_episode = e
    state = env.reset()
    # First element of state is the observation in a numpy array list.
    state = state[0]
    # PyTorch performs best when the input is a single numpy array, not a list of numpy arrays
    state = np.array(state)
    while True:
        action = mario.act(state) # Get action
        next_state, reward, done, truncated, info = env.step(action) # Take action
        next_state = np.array(next_state) # Convert to numpy array
        mario.cache(state, next_state, action, reward, done) # Cache the experience
        q, loss = mario.learn() # Train the agent
        logger.log_step(reward, loss, q, info['x_pos']) # Log progress
        state = next_state # Update state

        if done or info['flag_get']: # If episode is over, reset
            break
    # metrics.py will automatically plot the progress
    logger.log_episode()

    # Log every 50 episodes, but don't save until 100
    if e % 50 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
