from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
for step in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
            state = env.reset()
print(f"ACTIONS:\n{env.action_space}\n\n")
print(f"ACTION MEANINGS:\n{env.get_action_meanings}\n\n")

env.close()

