from cv_agent import CVAgent
from time import time



# ranges to test for each parameter
# start (inc), stop (exc), steps
# SPA_test_range = (4, 9, 1) # start at 4 otherwise mario never gets over pipes
# GOOMBA_test_range = (30,81,5)
# KOOPA_test_range = (30,81,5)
SPA_test_range = (7, 9, 1) # start at 4 otherwise mario never gets over pipes
GOOMBA_test_range = (30,81,5)
KOOPA_test_range = (30,81,5)


runs_per_spa = len(range(GOOMBA_test_range[0], GOOMBA_test_range[1], GOOMBA_test_range[2])) * len(range(KOOPA_test_range[0], KOOPA_test_range[1], KOOPA_test_range[2]))

if(__name__ == "__main__"):
    agent = CVAgent(debug=None)
    with open('./experiment-data/experiment_dump.tsv', 'w') as all:
        with open('./experiment-data/averages_dump.tsv', 'w') as avg:
            all.write("STEPS_PER_ACTION\tGOOMBA_RANGE\tKOOPA_RANGE\tRUN_SCORE\tRUN_TIME\tSTEPS\n")
            avg.write("STEPS_PER_ACTION\tAVG_RUN_SCORE\tAVG_RUN_TIME\tAVG_STEPS\n")

            for spa in range(SPA_test_range[0], SPA_test_range[1], SPA_test_range[2]):
                starts_spa = time()
                averages_spa = {
                    'run-score-avg': 0,
                    'run-time-avg': 0,
                    'steps-avg': 0
                }
                for goomba in range(GOOMBA_test_range[0], GOOMBA_test_range[1], GOOMBA_test_range[2]):
                    for koopa in range(KOOPA_test_range[0], KOOPA_test_range[1], KOOPA_test_range[2]):
                        agent.STEPS_PER_ACTION = spa
                        agent.GOOMBA_RANGE = goomba
                        agent.KOOPA_RANGE = koopa
                        print(f"Running test for metrics:\n\tSPA: {spa}\n\tGOOMBA_RANGE: {goomba}\n\tKOOPA_RANGE: {koopa}\n\n")
                        data = agent.play(metrics=True)
                        if data == None: 
                            continue
                        averages_spa['run-score-avg'] += data['run-score']
                        averages_spa['run-time-avg'] += data['run-time']
                        averages_spa['steps-avg'] += data['steps']

                        all.write(f"{spa}\t{goomba}\t{koopa}\t{data['run-score']}\t{data['run-time']}\t{data['steps']}\n")
                end_spa = time() - starts_spa
                print(f"Time taken to run tests on {spa} steps per action: {round(end_spa,2)}s")

                for average in averages_spa.items():
                    averages_spa[average[0]] = average[1] / runs_per_spa

                avg.write(f"{spa}\t{round(averages_spa['run-score-avg'], 4)}\t{round(averages_spa['run-time-avg'],4)}\t{round(averages_spa['steps-avg'], 4)}\n")


    print("Experiment complete, data written to file")