import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, ClipScoreboardWrapper, RewardShaper
import numpy as np
import cv2

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
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
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints1') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('mario_net_1.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()[0]
    state = np.array(state)
    while True:

        frame = np.ascontiguousarray(env.render())

        action = mario.act(state)

        next_state, reward, done, truncated, info = env.step(action)
        metrics_space = np.zeros((150, 256, 3), dtype=np.uint8)
        cv2.putText(metrics_space, f"Action: {action}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(metrics_space, f"Epsilon: {mario.exploration_rate:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(metrics_space, f"Episode: {e}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(metrics_space, f"Reward: {logger.curr_ep_reward:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(metrics_space, f"X-Pos: {info['x_pos']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(metrics_space, f"Step: {mario.curr_step}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        combined_frame = np.vstack((metrics_space, frame))
        cv2.imshow('Super Mario Bros with Overlays', combined_frame)
        cv2.waitKey(1)
       # reward = RewardShaper().get_reward(reward, info, done)
        next_state = np.array(next_state)
        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None, info['x_pos'])

        state = next_state

        if done or info['flag_get']:
            break
    cv2.destroyAllWindows()
    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )