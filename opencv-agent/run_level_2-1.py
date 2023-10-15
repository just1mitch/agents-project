from cv_agent import CVAgent
import numpy as np
from pathlib import PurePath

## Take the run parameters of iterations
## that beat level 1-1 and run them on 2-1
if(__name__ == "__main__"):
    run_list = np.loadtxt(PurePath("opencv-agent/experiment-data/successful_1-1.tsv"),
               skiprows=1,
               usecols=(0,1,2),
               delimiter='\t',
               converters=float,
               dtype=[
                                ('SPA', np.int_), 
                                ('GOOMBA', np.int_),
                                ('KOOPA', np.int_),
                            ])
    with open(PurePath("opencv-agent/experiment-data/level_2-1_dump.tsv"), 'w') as file:
        file.write("STEPS_PER_ACTION\tGOOMBA_RANGE\tKOOPA_RANGE\tRUN_SCORE\tRUN_TIME\tSTEPS\tX_POS\n")
        agent = CVAgent()
        for run in run_list:
            agent.__init__(debug=None, level='2-1')
            agent.STEPS_PER_ACTION = run[0]
            agent.GOOMBA_RANGE = run[1]
            agent.KOOPA_RANGE = run[2]

            metrics = agent.play(metrics=True)
            print(f"{run[0]}\t{run[1]}\t{run[2]}\t{metrics['run-score']}\t{metrics['run-time']}\t{metrics['steps']}\t{metrics['x_pos']}\n")
            file.write(f"{run[0]}\t{run[1]}\t{run[2]}\t{metrics['run-score']}\t{metrics['run-time']}\t{metrics['steps']}\t{metrics['x_pos']}\n")
