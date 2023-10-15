from csv import reader
from matplotlib import pyplot as plt
import numpy as np
from pathlib import PurePath

path = PurePath("opencv-agent/experiment-data/level_1-1_dump.tsv")

# Get all runs from the list of iterations where reward
# score is over 3000 (associated with a level completion)
def get_successful_runs(datalist):
    return datalist[np.where(datalist['RUN_SCORE'] > 3000.0)]

# Parse a tab separated file for the data
def get_datalist(file):
    with open(file, 'r') as datafile:
        data = reader(datafile, delimiter="\t")
        datalist = []
        next(data)
        for line in data:
            for i, datapoint in enumerate(line):
                line[i] = round(float(datapoint), 3)
            datalist.append(tuple(line))
        
        datalist = sorted(datalist, key=lambda x:x[4])
        datalist = np.array(datalist, 
                            dtype=[
                                ('SPA', np.int_), 
                                ('GOOMBA', np.int_),
                                ('KOOPA', np.int_),
                                ('RUN_SCORE', np.int_), 
                                ('RUN_TIME', np.float_),
                                ('STEPS', np.int_)
                            ])
    return datalist

if(__name__ == "__main__"):
    
    datalist = get_datalist(path)

    # Data analysis on winning runs (where reward is greater than 3000)
    successful_runs = get_successful_runs(datalist)
    np.savetxt(PurePath("opencv-agent/experiment-data/successful_1-1.tsv"),
               X=successful_runs,
               delimiter="\t", 
               newline="\n",
               comments="",
               header="STEPS_PER_ACTION\tGOOMBA_RANGE\tKOOPA_RANGE\tRUN_SCORE\tRUN_TIME\tSTEPS")

    print(f"Number of successful runs: {len(successful_runs)}/{len(datalist)} ({round(len(successful_runs)/len(datalist), 2)}%)")
    spa_sorted = sorted(successful_runs, key=lambda x:x[0])
    print(f"Runs were successful with a STEPS_PER_ACTION value between {int(spa_sorted[0][0])} and {int(spa_sorted[-1][0])}")

    goomba_sorted = sorted(successful_runs, key=lambda x:x[1])
    print(f"Runs were successful with a GOOMBA_RANGE value between {int(goomba_sorted[0][1])} and {int(goomba_sorted[-1][1])}")

    koopa_sorted = sorted(successful_runs, key=lambda x:x[2])
    print(f"Runs were successful with a KOOPA_RANGE value between {int(koopa_sorted[0][2])} and {int(koopa_sorted[-1][2])}\n")

    # Fastest run sorts successful runs by their time first, then their reward score second
    fastest_run_ind = np.lexsort((-successful_runs['RUN_SCORE'], successful_runs['RUN_TIME']))[0]
    fastest_run = successful_runs[fastest_run_ind]
    print(f"Fastest run parameters:\nTime to completion: {fastest_run[4]}s\nSTEPS_PER_ACTION: {fastest_run[0]}\nGOOMBA_RANGE: {fastest_run[1]}\nKOOPA_RANGE: {fastest_run[2]}\n")

    # Highest score sorts successful runs by their score first, then their time second
    highest_score_ind = np.lexsort((successful_runs['RUN_TIME'], -successful_runs['RUN_SCORE']))[0]
    highest_score = successful_runs[highest_score_ind]
    print(f"Highest score parameters:\nReward score: {highest_score[3]}\nSTEPS_PER_ACTION: {highest_score[0]}\nGOOMBA_RANGE: {highest_score[1]}\nKOOPA_RANGE: {highest_score[2]}\n")

    # Scatter plot of successful runs
    params = ['SPA', 'GOOMBA', 'KOOPA']
    time = [tup[4] for tup in successful_runs]
    reward = [tup[3] for tup in successful_runs]
    colours = ['red' if i in [fastest_run_ind, highest_score_ind] else 'blue' for i in range(len(successful_runs))]
    plt.figure(figsize=(10, 8))
    plt.scatter(time, reward, marker='.', c=colours)
    offset = ((plt.xlim()[1] - plt.xlim()[0])/50, (plt.ylim()[1] - plt.ylim()[0])/75)
    plt.annotate(f"{[fastest_run[param] for param in params]}", (time[fastest_run_ind], reward[fastest_run_ind]), xytext=(time[fastest_run_ind]+offset[0], reward[fastest_run_ind]+offset[1]), fontsize = 10)
    plt.annotate(f"{[highest_score[param] for param in params]}", (time[highest_score_ind], reward[highest_score_ind]), xytext=((time[highest_score_ind]+offset[0]), reward[highest_score_ind]+offset[1]), fontsize = 10)
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title("Performance of OpenCV Agent in Successful Super Mario Bros Level 1-1 Iterations", loc='center', pad=1.5)
    plt.figtext(0.5, 0.01, "Tests ran on Lenovo Legion 5i, Intel(R) Core(TM) i5-10300H CPU, NVIDIA GeForce GTX 1650Ti\nSee README for testing details", wrap=True, horizontalalignment='center')
    figpath = PurePath("opencv-agent/experiment-data/successful_1-1_runs.png")
    plt.savefig(figpath)
    print(f"Analysis Complete, plot saved to {figpath}")
    
    # Scatter plot of all data
    time = [tup[4] for tup in datalist]
    reward = [tup[3] for tup in datalist]
    plt.figure(figsize=(10,8))
    plt.scatter(time, reward, marker='.')
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.figtext(0.5, 0.01, "Tests ran on Lenovo Legion 5i, Intel(R) Core(TM) i5-10300H CPU, NVIDIA GeForce GTX 1650Ti\nSee README for testing details", wrap=True, horizontalalignment='center')
    plt.title("Performance of OpenCV Agent in all Super Mario Bros Level 1-1 Iterations", loc='center', pad=1.5)
    figpath = PurePath("opencv-agent/experiment-data/all_1-1_runs.png")
    plt.savefig(figpath)

