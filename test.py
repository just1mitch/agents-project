from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import matplotlib.pyplot as plt

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
#env = gym.wrappers.FrameStack(env, num_stack=4)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
marios_actions = [1, 1, 4, 1, 1, 1, 1, 4, 1]
for step in range(5000):
    action = marios_actions[step % len(marios_actions)]
    obs, reward, terminated, truncated, info = env.step(action)
    print("step", step, "action", action, "reward", reward, "terminated", terminated, "truncated", truncated, "info", info)
    print("Shape of ObState", obs.shape)
    print(obs)
    done = terminated or truncated
    state = env.reset()
    plt.imshow(state)    
    print(input ("Press Enter to continue..."))

env.close()
