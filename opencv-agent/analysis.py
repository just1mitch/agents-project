from csv import reader
from matplotlib import pyplot as plt
import numpy as np
from pathlib import PurePath

datafile = PurePath("./experiment-data/experiment_dump_all.tsv")
averages = PurePath("./exeriment-data/averages_dump_all.tsv")

if(__name__ == "__main__"):
    with open(datafile, 'r') as file:
        data = reader(file, delimiter="\t", )
        datalist = []
        next(data)
        for line in data:
            for i, datapoint in enumerate(line):
                line[i] = round(float(datapoint), 3)
            datalist.append(tuple(line))
        
        datalist = sorted(datalist, key=lambda x:x[4])
        datalist = np.array(datalist)

    # Data analysis on winning runs (where reward is greater than 3000)
    successful_runs = datalist[np.where(datalist[:,3] > 3000.0)]

    print(f"Number of successful runs: {len(successful_runs)}/{len(datalist)} ({round(len(successful_runs)/len(datalist), 2)}%)")
    spa_sorted = sorted(successful_runs, key=lambda x:x[0])
    print(f"Runs were successful with a STEPS_PER_ACTION value between {int(spa_sorted[0][0])} and {int(spa_sorted[-1][0])}")

    goomba_sorted = sorted(successful_runs, key=lambda x:x[1])
    print(f"Runs were successful with a GOOMBA_RANGE value between {int(goomba_sorted[0][1])} and {int(goomba_sorted[-1][1])}")

    koopa_sorted = sorted(successful_runs, key=lambda x:x[2])
    print(f"Runs were successful with a KOOPA_RANGE value between {int(koopa_sorted[0][2])} and {int(koopa_sorted[-1][2])}\n")

    # Fastest run sorts successful runs by their time first, then their reward score second
    fastest_run_ind = np.lexsort((-successful_runs[:,3], successful_runs[:,4]))[0]
    fastest_run = successful_runs[fastest_run_ind]
    print(f"Fastest run parameters:\nTime to completion: {fastest_run[4]}s\nSTEPS_PER_ACTION: {fastest_run[0]}\nGOOMBA_RANGE: {fastest_run[1]}\nKOOPA_RANGE: {fastest_run[2]}\n")

    # Highest score sorts successful runs by their score first, then their time second
    highest_score_ind = np.lexsort((successful_runs[:,4], -successful_runs[:,3]))[0]
    highest_score = successful_runs[highest_score_ind]
    print(f"Highest score parameters:\nReward score: {highest_score[3]}\nSTEPS_PER_ACTION: {highest_score[0]}\nGOOMBA_RANGE: {highest_score[1]}\nKOOPA_RANGE: {highest_score[2]}\n")

    # Show scatter plot of successful runs
    time = [tup[4] for tup in successful_runs]
    reward = [tup[3] for tup in successful_runs]
    colours = ['red' if i in [fastest_run_ind, highest_score_ind] else 'blue' for i in range(len(successful_runs))]
    plt.scatter(time, reward, marker='.', c=colours)
    offset = ((plt.xlim()[1] - plt.xlim()[0])/50, (plt.ylim()[1] - plt.ylim()[0])/75)
    plt.annotate(f"{fastest_run[:3]}", (time[fastest_run_ind], reward[fastest_run_ind]), xytext=(time[fastest_run_ind]+offset[0], reward[fastest_run_ind]+offset[1]), fontsize = 6)
    plt.annotate(f"{highest_score[:3]}", (time[highest_score_ind], reward[highest_score_ind]), xytext=((time[highest_score_ind]+offset[0]), reward[highest_score_ind]+offset[1]), fontsize = 6)
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title("Time vs Reward Graph for Successful Super Mario Bros Iterations", loc='center', pad=1.5)
    figpath = PurePath("./experiment-data/successful-runs.png")
    plt.savefig(figpath)
    print(f"Analysis Complete, plot saved to {figpath}")