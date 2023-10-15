import gym
import torch
from stable_baselines3 import PPO

# Unused / untested in this project - but may be useful in the future

# Load the pre-trained models
model1 = PPO.load('./train/best_model.zip')
model2 = PPO.load('./train/best_model_500000.zip')

# Define the weights for combining the models
weight_model1 = 0.6
weight_model2 = 0.4

# Get the policy parameters from each model
policy_params1 = model1.policy.state_dict()
policy_params2 = model2.policy.state_dict()

# Combine the policy parameters using the specified weights
combined_policy_params = {}
for param_name in policy_params1.keys():
    combined_policy_params[param_name] = (
        weight_model1 * policy_params1[param_name] +
        weight_model2 * policy_params2[param_name]
    )

# Create a new PPO model with the same environment as model1
combined_model = PPO('CnnPolicy', model1.env)

# Load the combined policy parameters into the new model
combined_model.policy.load_state_dict(combined_policy_params)

# Save the combined model
combined_model.save("./train/combined_model_best.zip")
