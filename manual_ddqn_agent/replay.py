import random, datetime
from pathlib import Path
import os
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, ClipScoreboardWrapper, XValueRewardWrapper
import numpy as np
import cv2
import argparse

# Replay the trained model with a specified checkpoint using the --checkpoint flag
# Much of the replay is derrived from: https://github.com/yfeng997/MadMario/blob/master/replay.py
# Additional code is added to display the metrics on the screen and to display the current action being taken by the agent

# For further information on the setup of replay.py, please refer to main.py as this is a modified version of the main.py file

def displayMetrics(frame):
    # Create a blank space to display the metrics
    metrics_space = np.zeros((150, 256, 3), dtype=np.uint8)
    cv2.putText(metrics_space, f"Action: {action}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(metrics_space, f"Epsilon: {mario.exploration_rate:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(metrics_space, f"Episode: {e}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(metrics_space, f"Reward: {logger.curr_ep_reward:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(metrics_space, f"X-Pos: {info['x_pos']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(metrics_space, f"Step: {mario.curr_step}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    combined_frame = np.vstack((metrics_space, frame))
    return combined_frame

def log_to_file(log_filename, reward, steps, episode):
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{episode},{reward},{steps}\n")
parser = argparse.ArgumentParser(description="Run the Mario agent with a specified checkpoint.")
parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file.")
parser.add_argument('--render', action='store_true', help="Show a Model in action using render output.")
args = parser.parse_args()

RENDER = args.render

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array') # Set render_mode to rgb_array to get the rendered frames
env = ClipScoreboardWrapper(env)
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = XValueRewardWrapper(env)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpointsReplay') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path(args.checkpoint)
log_filename = os.path.splitext(checkpoint)[0] + "_log.txt"
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

# Set the exploration rate to min so that the agent will rely on it's policy to exploit
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

# Run the agent for 100 episodes
episodes = 100
crossed_1200_count = 0
total_steps_to_cross_1200 = 0
for e in range(episodes):

    state = env.reset()[0]
    state = np.array(state)
    steps = 0
    flag_get_message_printed = False
    crossed_1200_message_printed = False
    
    while True:
        # env.render() - cv2.imshow() is used instead to display the metrics on the screen

        action = mario.act(state)

        next_state, reward, done, truncated, info = env.step(action)
        if RENDER: 
            frame = np.ascontiguousarray(env.render())
            cv2.imshow('Super Mario Bros with Overlays', displayMetrics(frame))
            cv2.waitKey(1)
        steps += 1
        total_steps_to_cross_1200 += steps
        next_state = np.array(next_state)
        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None, info['x_pos'])

        state = next_state

        if info['flag_get'] and not flag_get_message_printed:
                print(f"Model {checkpoint} completed the level in episode {e+1} with {steps} steps.")
                flag_get_message_printed = True
                
            # Check if Mario gets past 1200
        elif info['x_pos'] > 1200 and not crossed_1200_message_printed:
            crossed_1200_count += 1
            print(f"Model {checkpoint} crossed 1200 in episode {e+1} with {steps} steps.")
            crossed_1200_message_printed = True

        if done or info['flag_get']:
            break
    if RENDER:
        cv2.destroyAllWindows()
    logger.log_episode()
    log_to_file(log_filename, info['x_pos'], steps, e+1)
    if e % 1 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
average_steps_to_cross_1200 = total_steps_to_cross_1200 / crossed_1200_count if crossed_1200_count != 0 else 0
print(f"DDQN Model {checkpoint} surpassed x-position 1200 {crossed_1200_count} times over {episodes} episodes.")
print(f"DDQN Model {checkpoint} took an average of {average_steps_to_cross_1200:.2f} steps to surpass x-position 1200.")
