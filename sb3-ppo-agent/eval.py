import tkinter as tk
from tkinter import filedialog
import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
import cv2
import os
import argparse
from PIL import Image
parser = argparse.ArgumentParser(description="Mario SB3-PPO Evaluator")
parser.add_argument('--render', action='store_true', help="Show a Model in action using render output.")
args = parser.parse_args()

# Evalate a model or series of models using the specified number of episodes
# Perform a render if the --render flag is set - so we can see the model in action
# Displays metrics such as the action taken, the model name, the episode number, the step number, the x-position, the learning rate and the reward

# Additional logging was done for comparison with other agents.
# Uses tk for a file dialog to select the model files to evaluate - this is a bit easier than typing the full path like in manual-ddqn-agent

RENDER = args.render
# Wrapper to remove seed and options from reset
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
    
def log_to_file(log_filename, reward, steps, episode):
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{episode},{reward},{steps}\n")

def evaluate_model(file_path, episodes=75, visible=False):
    """Evaluate a model using the specified number of episodes"""
    # Create a standard Super Mario Bros environment - should be the same as the one used for training
    env = gym.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="rgb_array")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RemoveSeedWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = CustomReshapeAndResizeObs(env, shape=(84, 84))
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    # Should match the env used for training - we can monitor the env w a Wrapper
    #env = VecMonitor(env, filename=None, keep_buf=100)
    model = PPO.load(file_path)
    total_distance = 0
    model_filename = os.path.basename(file_path)
    log_filename = os.path.splitext(file_path)[0] + "_log.txt"
    frames = [] # Used for creating a GIF
    crossed_1200_count = 0
    total_steps_to_cross_1200 = 0
    for episode in range(episodes):
        # Reset the environment and get the initial observation for a run
        obs = env.reset()
        done = False
        steps = 0
        flag_get_message_printed = False
        crossed_1200_message_printed = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if visible:
                frame = np.ascontiguousarray(env.render())
                
                # Add some overlay text to the frame
                metrics_space = np.zeros((150, 256, 3), dtype=np.uint8)
                cv2.putText(metrics_space, f"Action: {action}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(metrics_space, f"Model: {model_filename}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(metrics_space, f"Episode: {episode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(metrics_space, f"Step: {steps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(metrics_space, f"X-Pos: {info[0]['x_pos']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(metrics_space, f"Learning Rate: {model.lr_schedule(1):.2e}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(metrics_space, f"Reward: {reward}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                combined_frame = np.vstack((metrics_space, frame))
                            
                cv2.imshow('Super Mario Bros with Overlays', combined_frame)
                frames.append(combined_frame)
                cv2.waitKey(1)
            steps += 1

            # Check if Mario completes the level using flag_get
            if info[0]['flag_get'] and not flag_get_message_printed:
                print(f"Model {file_path} completed the level in episode {episode+1} with {steps} steps.")
                flag_get_message_printed = True
                
            # Check if Mario gets past 1200
            elif info[0]['x_pos'] > 1200 and not crossed_1200_message_printed:
                crossed_1200_count += 1
                total_steps_to_cross_1200 += steps
                print(f"Model {file_path} crossed 1200 in episode {episode+1} with {steps} steps.")
                crossed_1200_message_printed = True

        total_distance += info[0]['x_pos']
        log_to_file(log_filename, info[0]['x_pos'], steps, episode+1)
        if visible: cv2.destroyAllWindows()
    average_steps_to_cross_1200 = total_steps_to_cross_1200 / crossed_1200_count if crossed_1200_count != 0 else 0
    env.close()
    return total_distance / episodes, frames, crossed_1200_count, average_steps_to_cross_1200


def load_and_evaluate_models():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(filetypes=[("ZIP files", "*.zip")])
    
    if not file_paths:
        print("No model files selected. Exiting.")
        return

    best_distance = 0
    best_model_path = ""
    
    for file_path in file_paths:
        avg_distance, frames, crossed_1200_count, average_steps_to_cross_1200 = evaluate_model(file_path, visible=RENDER)
        print(f"Model {file_path} achieved an average distance of {avg_distance:.2f}")
        print(f"PPO Model {file_path} surpassed x-position 1200 {crossed_1200_count} times over {100} episodes.")
        print(f"PPO Model {file_path} took an average of {average_steps_to_cross_1200:.2f} steps to surpass x-position 1200.")
        if avg_distance > best_distance:
            best_distance = avg_distance
            best_model_path = file_path
        
        if RENDER and frames:
            gif_path = os.path.splitext(file_path)[0] + ".gif"
            frames_pil = [Image.fromarray(frame) for frame in frames]
            frames_pil[0].save(gif_path, save_all=True, append_images=frames_pil[1:], loop=0, duration=40)
            print(f"Gameplay saved as {gif_path}")

    print(f"\nThe best model is {best_model_path} with an average distance of {best_distance:.2f}")

if __name__ == "__main__":
    load_and_evaluate_models()
