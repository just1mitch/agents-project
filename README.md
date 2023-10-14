
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
  
<div class="row">
  <div class="column">
    <img src="opencv-agent/report-data/Mario%20Gif.gif" alt="Mario Gif" width="200" height="200">
  </div>
  <div class="column">
    <img src="opencv-agent/report-data/MarioDetect Gif.gif" width="200" height="200">
  </div>
</div>

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

**Running the Agent:**
1. Navigate to manual_agent folder
2. Check `main.py` and the `RESUME` flag, `True = Continued Training, False = New Training`
- If you are training a new model, ensure `RESUME = False`
- Resumed-Training will always presume the latest CheckPoint and continue from latest log.
- Resumed-Training will carry over episodes, steps and current Epsilon (Exploration Rate).
4. Run `python main.py`

The Agent will then log to a Time Stamped Directory and the Super Mario Environment will begin training.
A Model Checkpoint will be saved every `50,000` steps.

<div class="row">
  <div class="column">
    <img src="manual_ddqn_agent/pictures/gameplay.gif" alt="Mario Gif" width="256" height="390">
  </div>
    
**Evaluating a Model:**
1. Navigate to manual_agent folder
2. Run 'python review.py`` --checkpoint CHECKPOINT_NAME``
   
This will create a Tester Environment and an additional logging directory. It will attempt to use the specified model with an ``Epsilon`` of `0.1` (Indicating defined action and no Exploration)

