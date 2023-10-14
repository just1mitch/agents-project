
# CITS3001 - Algorithms, Agents and AI Project

### Authors:
- Mitchell Otley (23475725)
- Jack Blackwood (23326698)

#### *Please refer to the project report for breakdown and analysis of the DQNN and OpenCV agents. This README purely describes how to use the agents*

## Create the conda environment
1. Navigate to root directory (where environment.yml file is located)
2. In a conda shell, run `conda env create -f environment.yml`
3. Once environment is created, run `conda activate mario`
4. Follow steps to run the agents from within the environment

## **OpenCV Agent**

**Running the Agent:**
1. Navigate to opencv-agent folder
2. Edit values inside `cv_agent.py` file to change how agent plays 
3. Run `python cv_agent.py`


Variables available to change are:
- `CVAgent(debug=[None, 'console', 'detect'])`  
    None - No debugging  
    'console' - Show console messages  
    'detect' - Show detection screen and console messages

<img src="opencv-agent/report-data/Mario%20Gif.gif" width="200" height="200">
<img src="opencv-agent/report-data/MarioDetect Gif.gif" width="200" height="200">  

- `agent.STEPS_PER_ACTION`  
    Number of steps taken before another action is chosen

- `agent.GOOMBA_RANGE`  
    Range (in pixels) between Mario and a Goomba before Mario will jump

- `agent.KOOPA_RANGE`  
    Range (in pixels) between Mario and a Koopa before Mario will jump

- `agent.play(metrics=[False, True])`  
    - False - Return `None` when iteration is finished  
    - True - Return a dictionary when iteration is finished:  
            'run-score': total score of iteration,  
            'run-time': time to complete iteration,  
            'steps': steps taken in the iteration 

___
**Run Analysis**

To see an analysis of the experiment data, run `python analysis.py`


## **DQNN Agent**

[insert text here]
