import numpy as np
from pathlib import PurePath
from matplotlib import pyplot as plt

## Generate a scatter plot comparison of all 3 Agents on Level 2-1

# OpenCV Agent data
opencvpath = PurePath("opencv-agent/experiment-data/level_2-1_dump.tsv")
opencv_data = np.loadtxt(opencvpath,
                         skiprows=1,
                         usecols=(3, 5),
                         delimiter='\t',
                         converters=float,
                         dtype= [('Reward', np.int_), ('Steps', np.int_)]
                         )

# DDQN Agent data
ddqnpath = PurePath("opencv-agent/experiment-data/ddqn_2-1_dump.txt")
ddqn_data = np.loadtxt(ddqnpath,
                       skiprows=1,
                       usecols=(1,2),
                       delimiter=',',
                       converters=float,
                       dtype=[('Reward', np.int_), ('Steps', np.int_)]
                       )

# PPO Agent data
ppopath = PurePath("opencv-agent/experiment-data/ppo_2-1_dump.txt")
ppo_data = np.loadtxt(ppopath,
                       skiprows=1,
                       usecols=(1,2),
                       delimiter=',',
                       converters=float,
                       dtype=[('Reward', np.int_), ('Steps', np.int_)]
                       )

x_range = np.array(range(0, max(len(opencv_data), len(ddqn_data), len(ppo_data)), 1))
names = ["OpenCV Agent", "DDQN Agent", "PPO Agent"]

# Uncomment for log of reward/step
fig, axes = plt.subplots()
for i, agent in enumerate([opencv_data, ddqn_data, ppo_data]):
    axes.scatter(np.log(agent['Steps']), np.log(agent['Reward']), label=f"{names[i]}", marker='.')
plt.ylabel("(Log of) Reward for Run")
plt.xlabel("(Log of) Steps Taken")
plt.title("Reward per Step for 75 Iterations of Different Agents on Level 2-1", loc='center', pad=5)
plt.legend()
plt.savefig(PurePath("opencv-agent/experiment-data/level_2-1_comparison.png"))

# # Uncomment for actual reward/step
# fig, axes = plt.subplots()
# for i, agent in enumerate([opencv_data, ddqn_data, ppo_data]):
#     axes.scatter(agent['Steps'], agent['Reward'], label=f"{names[i]}", marker='.')
# plt.ylabel("Reward for Run")
# plt.xlabel("Steps Taken")
# plt.title("Reward per Step for 75 Iterations of Different Agents on Level 2-1", loc='center', pad=5)
# plt.legend()
# plt.show()

