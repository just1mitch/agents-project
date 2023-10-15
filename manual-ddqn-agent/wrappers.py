import gym
import numpy as np
from skimage import transform
from gym.spaces import Box

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        # We need to make sure it is a Box space since we use np.array() later
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return (transform.resize(observation, self.shape) * 255).astype(np.uint8)
    
#Unused reshape and normalise wrapper - was required for sbo-ppo-agent-v2 - this combines resize and normalise via transform observation

# class CustomReshapeAndResizeObs(gym.ObservationWrapper):
#     def __init__(self, env, shape=(84, 80)):
#         super(CustomReshapeAndResizeObs, self).__init__(env)
#         old_shape = self.observation_space.shape
#         self.shape = shape
#         self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, shape[0], shape[1]), dtype=np.float32)  # Adjusted channel dimension

#     def observation(self, observation):
#         observation = observation[32:, :]
#         # Resize the observation
#         observation = cv2.resize(observation, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)  # Note the order of dimensions

#         # Normalize
#         observation = observation.astype(np.float32) / 255.0
#         # Add channel dimension
#         observation = np.expand_dims(observation, axis=0)
#         return observation

# class EnsureNumpyArray(gym.ObservationWrapper):
#     def observation(self, observation):
#         return np.array(observation)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info

# This is used to remove the seed from the environment reset, as it is not compatible with current gym version
class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)
    
# Experimental reward shaper - not used in final agent
# # This would shape the reward outside of the environment, but it is better to do it inside the environment
# class RewardShaper:
#     def __init__(self):
#         self.prev_x_pos = 0
#         self.stuck_counter = 0
#         self.prev_enemy_kills = 0

#     def get_reward(self, reward, info, done):
#         # Check if Mario's x-position increased
#         if info['x_pos'] > self.prev_x_pos:
#             self.prev_x_pos = info['x_pos']
#             self.stuck_counter = 0  # Reset stuck counter
#         else:
#             self.stuck_counter += 1

#         # Check if Mario killed any enemies
#         if info['score'] > self.prev_enemy_kills:
#             reward += 2.0  # Bonus for killing an enemy
#             self.prev_enemy_kills = info['score']

#         # Penalty if Mario is stuck
#         if self.stuck_counter > 80:  # Assume stuck if no progress for 80 steps
#             reward -= 1.0
#             self.stuck_counter = 0  # Reset stuck counter

#         # Large penalty for death
#         if done and not info['flag_get']:
#             reward -= 5

#         # Bonus for level completion
#         if info['flag_get']:
#             reward += 10

#         return reward

# Experimental reward shaper - not used in final agent
# class EnhancedRewardWrapper(gym.Wrapper):
#     def __init__(self, env, new_max_x_bonus=5, stuck_penalty=-3, stuck_threshold=100):
#         super(EnhancedRewardWrapper, self).__init__(env)
#         self.max_x_pos = float('-inf')  # Keep track of the maximum x position reached
#         self.last_x_pos = None  # Keep track of the last x position
#         self.stuck_counter = 0  # Counter to check if Mario is stuck
#         # Bonus for reaching a new maximum x position
#         self.new_max_x_bonus = new_max_x_bonus
#         # Penalty for staying in the same x position for too long
#         self.stuck_penalty = stuck_penalty
#         # Number of frames to consider Mario as stuck
#         self.stuck_threshold = stuck_threshold

#     def reset(self, **kwargs):
#         obs = super().reset(**kwargs)
#         self.max_x_pos = float('-inf')
#         self.last_x_pos = None
#         self.stuck_counter = 0
#         return obs

#     def step(self, action):
#         obs, reward, done, truncated, info = super().step(action)

#         x_pos = info['x_pos'] 

#         # Bonus for new max x position
#         if x_pos > self.max_x_pos:
#             reward += self.new_max_x_bonus
#             self.max_x_pos = x_pos

#         # Check if Mario is stuck
#         if x_pos == self.last_x_pos:
#             self.stuck_counter += 1
#             if self.stuck_counter >= self.stuck_threshold:
#                 reward += self.stuck_penalty
#                 self.stuck_counter = 0  # Reset counter after applying penalty
#         else:
#             self.stuck_counter = 0  # Reset counter if Mario moved

#         self.last_x_pos = x_pos  # Update the last x position for the next step

#         return obs, reward, done, truncated, info

# Simple reward shaper - used in final agent to give a slight extra reward for x-value progression
class XValueRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x_value = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        x_value_progress = info['x_pos'] - self.prev_x_value
        reward += x_value_progress * 0.003  # Scale the reward by a factor for x-value progression
        self.prev_x_value = info['x_pos']
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.prev_x_value = 0
        return self.env.reset(**kwargs) 
    
# Remove the top 32 pixels which contain the scoreboard, as it is not relevant to the agent, we don't want it to learn to read the score and associate with rewards
class ClipScoreboardWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Adjust observation space shape to account for the clipped scoreboard
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.env.observation_space.shape[0] - 32, self.env.observation_space.shape[1], self.env.observation_space.shape[2]), dtype=np.uint8)

    def observation(self, obs):
        return obs[32:]  # Remove the top 32 pixels which contain the scoreboard