import mario_locate_objects
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import time

# Main game playing agent
# Detection code adapted from 
# https://github.com/vmorarji/Object-Detection-in-Mario/blob/master/detect.py
class CVAgent:
    # A class holding information of the last state of the game
    # Allows heuristics to have a limited understanding 2of
    # the current occurrences of the game
    class LastState:
        last_action = None
        mario_world_coords = None
        mario_status = None
        enemy_locations = ()
        block_locations = ()
    
    last_state = LastState()
    env = None

    ## If DEBUG == None - no debugging
    ## If DEBUG == "console" - only console messages
    ## If DEBUG == "detect" - show detection screen and console messages
    DEBUG = None

    # Number of steps taken before another action is chosen
    STEPS_PER_ACTION = 8
    # Range (in pixels) between Mario and a Goomba before Mario will jump
    GOOMBA_RANGE = 45
    # Range (in pixels) between Mario and a Koopa before Mario will jump
    KOOPA_RANGE = 75
    # Range (in pixels) between Mario and a Koopa shell before Mario will jump
    SHELL_RANGE = 40
    
    jumping_hole = False # State of mario if he is jumping a hole
    jumping_enemy = False
    # Specifically for koopas - boolean is true if in loop for squashing koopa
    # Second value is a tracker of the sequence timeline
    jumping_koopa = (False, 0) 
    SEQ_LENGTH = 11

    
    def __init__(self, debug=None, level='1-1'):
        if debug not in [None, 'console', 'detect']:
            raise ValueError("""Invalid Debugging method, Please choose from:\n
                             None: No debugging\n
                             console: Console messages only\n
                             detect: Show detection boxes (significantly slower)\n""")
        self.DEBUG = debug
        if(self.DEBUG is not None): print(f"Running in debug mode: {self.DEBUG}")
        
        self.jumping_hole = False
        self.jumping_enemy = False
        self.jumping_koopa = (False, 0)

        levelname = f"SuperMarioBros-{level}-v0"
        if(self.DEBUG in ['detect']): self.env = gym.make(levelname, apply_api_compatibility=True)
        else: self.env = gym.make(levelname, apply_api_compatibility=True, render_mode="human")
        self.env = gym.wrappers.GrayScaleObservation(self.env)
        
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

    # Gets the block under mario, if there is one
    # if pipes is true, will check if mario standing on a pipe as well
    def __block_under_mario(self, block_locations, mario_location, pipes = True):
        if(pipes): blocks = [block for block in block_locations if (block[2] != 'pipe')]
        else: blocks = [block for block in block_locations]
        block_range = 5 # y-range from marios sprite to the block
        for block in blocks:
            if(abs(block[0][1] - mario_location[0][1] - mario_location[1][1]) in range(block_range)):
                if(mario_location[0][0] in range(block[0][0], block[0][0] + block[1][0])):
                    return block
        return None

    # Finds a hole in front of mario, if there is one
    def __find_hole(self, block_locations, mario_location):
        block = self.__block_under_mario(block_locations, mario_location, False)
        if(block != None):
            # found block mario is standing on, see if gap in immediate future
            gap_range = 48 # how many pixels in front of mario to check (48/16 = 3 blocks ahead)
            for i in range(0, gap_range, 16): # 16 is pixel size of blocks
                bricks = [brick[0] for brick in block_locations]
                if ((block[0][0] + i, block[0][1]) in bricks):
                    continue # found a block ahead
                else:
                    # hole found
                    if(self.DEBUG is not None): print(f"Hole found in front of mario {mario_location[0]}: {block[0]}")
                    return True
            # Found no holes, return empty list
            return False
        return False

    # special action sequence for koopas, to be able to shoot the shells forwards
    def __jump_koopa(self, state):
        if(self.jumping_koopa[0] == True):
            if(self.jumping_koopa[1] < self.SEQ_LENGTH - 6): # Frames 0-5
                # Phase 1: stopping momentum to jump on koopa
                self.jumping_koopa = (True, self.jumping_koopa[1]+1)
                action = 0
                return action
            elif (self.jumping_koopa[1] < self.SEQ_LENGTH - 2): # Frames 6-9
                # Phase 2: jumping on koopa
                self.jumping_koopa = (True, self.jumping_koopa[1]+1)
                action = 5
                return action
            elif(self.jumping_koopa[1] < self.SEQ_LENGTH): # Frame 10
                # Phase 3: shooting shell
                self.jumping_koopa = (True, self.jumping_koopa[1]+1)
                action = 6 # go left for a bit, to get behind shell
                return action
            else:
                self.jumping_koopa = (False, 0)
                return None
        return None

    # Given enemy locations, marios location and marios displacement, calculate
    # whether the enemy is a risk to marios life, and if so return True
    def __check_enemies(self, enemy_locations, mario_locations, mario_disp):
        for enemy in enemy_locations:
            # calculate the 'bottom left' corner of the sprites
            enemy_y = enemy[0][1] - enemy[1][1]
            mario_y = mario_locations[0][0][1] - mario_locations[0][1][1]
            y_range = 20 # how close in y-range the mario and enemy can be to detect they are on the same 'height'
            if(abs(enemy_y - mario_y) in range(y_range)):
                x_range = 10 # check 10 pixels left and right of previous frame to find enemy
                prev_enemies = self.last_state.enemy_locations
                for prev_enemy in prev_enemies: # search through previous frames enemies to find displacement
                    enemy_disp = (enemy[0][0] - prev_enemy[0][0], enemy[0][1] - prev_enemy[0][1])
                    if(prev_enemy[0][1] == enemy[0][1] and abs(enemy_disp[0]) <= x_range): # checks to see if previous enemy is the same as current
                        if(mario_disp[0] * enemy_disp[0] <= 0): # mario travelling towards enemy
                            if(enemy[2] == 'goomba'): jump_range = self.GOOMBA_RANGE # How close (in pixels) mario should get before jumping
                            elif(enemy[2] == 'koopa'): jump_range = self.KOOPA_RANGE    
                            elif(enemy[2] == 'koopashell'): 
                                jump_range = self.SHELL_RANGE # if koopa shell is returning to hit mario
                            if(enemy[0][0] - (mario_locations[0][0][0] + mario_locations[0][1][0]) in range(jump_range)):
                                if(enemy[2] == 'koopa'): self.jumping_koopa = (True, 0)
                                if(self.DEBUG is not None): print(f"Reacting to enemy {enemy[2]}: {enemy[0]}")
                                return True

    # Action function adapted from Lauren Gee's work
    # In mario_locate_objects.py
    def __make_action(self, screen, info, step, prev_action):

        # Secondary debugging screen to show openCV detection rectangles
        # Means that screen is scanned for opponents every frame
        if(self.DEBUG in ['detect']):
            mario_status = info["status"]
            object_locations = mario_locate_objects.locate_objects(screen, mario_status)

            mario_locations = object_locations["mario"]
            enemy_locations = object_locations["enemy"]
            block_locations = object_locations["block"]
            frame = cv.cvtColor(screen, cv.COLOR_RGB2BGR)
            if(enemy_locations):
                for enemy in enemy_locations:
                    pt1 = (enemy[0][0], enemy[0][1])
                    pt2 = (pt1[0] + enemy[1][0], pt1[1] + enemy[1][1])
                    cv.rectangle(frame, pt1, pt2, color=2)
                    cv.putText(frame, f"{enemy[2]}", (pt2[0]+5, pt1[1]), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=1)
            if(block_locations):
                for block in block_locations:
                    pt1 = (block[0][0], block[0][1])
                    pt2 = (pt1[0] + block[1][0], pt1[1] + block[1][1])
                    cv.rectangle(frame, pt1, pt2, color=2)
                    if(block[2] == 'question_block'): name = 'QB'
                    else: name = block[2]
                    # if block is not under mario, apply text

                    if(mario_locations and block[0][1] - mario_locations[0][0][1] < 0):
                        cv.putText(frame, name, (pt2[0], pt1[1]), color=(255,255,255), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3)
            cv.imshow("DEBUG: Detections", frame)
            if cv.waitKey(1)&0xFF == ord('q'): pass
        
        game_step = step % self.STEPS_PER_ACTION
        if game_step != 0 and game_step != self.STEPS_PER_ACTION // 2: 
            return prev_action
        # before next action is chosen, a snapshot of the game
        # is saved in a LastState object to be referenced when making
        # a choice on the next move
        elif(game_step == self.STEPS_PER_ACTION // 2):
            self.last_state.last_action = prev_action
            self.last_state.mario_world_coords = (info["x_pos"], info["y_pos"])

            # If debugging, some data has already been grabbed
            if self.DEBUG not in ['detect']:
                mario_status = info["status"]
                object_locations = mario_locate_objects.locate_objects(screen, self.last_state.mario_status)
            
            self.last_state.mario_status = mario_status

            # This is the format of the lists of locations:
            # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
            #
            # x_coordinate and y_coordinate are the top left corner of the object
            #
            # For example, the enemy_locations list might look like this:
            # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
            self.last_state.enemy_locations = object_locations["enemy"]
            # self.last_state.block_locations = object_locations["block"]

            # Grabbed the info to store for next action, now return prev_action
            return prev_action
        
        elif(game_step == 0):

            # already grabbed info if debugging
            if self.DEBUG not in ['detect']:
                mario_status = info["status"]
                object_locations = mario_locate_objects.locate_objects(screen, mario_status)

                mario_locations = object_locations["mario"]
                enemy_locations = object_locations["enemy"]
                block_locations = object_locations["block"]


            if(len(mario_locations) != 1):
                # Mario cannot be found, return action 1 to prevent errors
                action = 1
                return action
            

            
            # check for mario being on ground, as the sprites can be inconsistent
            block_under_mario = self.__block_under_mario(block_locations, mario_locations[0])
            if(block_under_mario != None): state = 'G'
            else: state = 'A'

            action = self.__jump_koopa(state)
            if(action != None): 
                return action # continue special action sequence for a koopa

            if(prev_action in [2,5] and state == 'G'): return 1 # run right to allow jump again

           
            # Mario displacement based on previous frames
            mario_disp = (info["x_pos"] - self.last_state.mario_world_coords[0], info["y_pos"] - self.last_state.mario_world_coords[1])
            # If mario on ground, follow decision tree
            if (state == 'G'):
                self.jumping_hole = False # no longer jumping if on ground again
                self.jumping_enemy = False

                enemy_found = self.__check_enemies(enemy_locations, mario_locations, mario_disp)
                if(enemy_found):
                    action = self.__jump_koopa(state)
                    if(action == None): action = 5
                    self.jumping_enemy = True
                    return action

                for block in block_locations:
                    # Check for pipes
                    if(block[2] == 'pipe'):
                        pipe_range = (10, 60) # How close (in pixels) mario should get before jumping over pipe - lowerbound, upperbound
                        mario_to_pipe = block[0][0] - mario_locations[0][0][0]
                        if(mario_to_pipe in range(pipe_range[0], pipe_range[1])):
                            # note - by doing (block - mario) we eliminate pipes that are to the left of mario
                            if(self.DEBUG is not None): print(f"Reacting to pipe: {block[0]}")
                            action = 2
                            return action
                    elif(enemy_locations == [] and block[2] == 'question_block'):
                        question_range = (10, 15) # how close mario should be to question block in x
                        mario_to_question = block[0][0] - mario_locations[0][0][0]
                        if(mario_to_question in range(question_range[0], question_range[1])):
                            if(mario_locations[0][0][1] - block[0][1] in range(10, 100)): # within y range
                                if(self.DEBUG is not None): print(f"Reacting to question block: {block[0]}")
                                return 2 # jump to hit question block
                    # Check if need to jump over a block
                    else:
                        block_range = (10, 30) # How close mario should get before jumping over block in path
                        if(block[0][0] - mario_locations[0][0][0] in range(block_range[0], block_range[1])):
                            # if halfway through mario cuts through block
                            if(mario_locations[0][0][1] + (mario_locations[0][1][1] / 2) in range(block[0][1], block[0][1] + block[1][1])):
                                if(self.DEBUG is not None): print(f"Reacting to block in front of mario: {block[0]}")
                                action = 2
                                return action
                        
                        
                # Check for hole in ground in front of mario
                hole = self.__find_hole(block_locations, mario_locations[0])
                if(hole):
                    self.jumping_hole = True
                    action = 2
                    return action

            # state: Air
            elif state == 'A':
                # jumping over hole
                if(self.jumping_hole): 
                    action = 2
                    return action
                
                # jumping over enemy
                if(self.jumping_enemy):
                    for enemy in enemy_locations:
                        if(abs(mario_locations[0][0][0] - enemy[0][0]) in range(self.GOOMBA_RANGE)):
                            if(mario_locations[0][0][1] - mario_locations[0][1][1] >= (enemy[0][1] + enemy[1][1])):
                                action = 2
                                return action

                if(mario_disp[1] > 0):
                    for block in block_locations:
                        # Check for pipes
                        if(block[2] == 'pipe'):
                            pipe_range = (10, 60) # How close (in pixels) mario should get before jumping over pipe - lowerbound, upperbound
                            mario_to_pipe = block[0][0] - mario_locations[0][0][0]
                            if(mario_to_pipe in range(pipe_range[0], pipe_range[1])):
                                if(mario_locations[0][0][1] + mario_locations[0][1][1] > block[0][1]):
                                    action = 2
                                else:
                                    action = 1
                                return action
                        # Add stuff for vertical blocks
                        else:
                            # Check to see if block is stopping mario from moving forwards
                            block_range = (10, 20) # How close the block is (x, y)
                            if(block[0][0] - mario_locations[0][0][0] in range(block_range[0], block_range[1])):
                                # if halfway through mario cuts through block
                                if(mario_locations[0][0][1] + (mario_locations[0][1][1] / 2) in range(block[0][1], block[0][1] + block[1][1])):
                                    action = 2
                                    return action
  
            action = 1
            return action

    # metrics param defines what data about a run to return to caller
    # A 'run' ends when mario has completed a level, or died
    # False = don't return anything
    # True = return metrics for run, including:
    #   - time taken to complete run (seconds)
    #   - reward score for run
    #   - number of steps in the run
    #   - global x_position reached
    def play(self, metrics=False):

        obs = None
        done = True
        self.env.reset()
        step = 0
        run_score = 0
        if(metrics): start = time.time_ns()

        # limited to 5000 steps
        while step < 5001:
            if obs is not None:
                action = self.__make_action(obs, info, step, action)
            else:
                action = 1
            obs, reward, terminated, truncated, info = self.env.step(action)
            run_score += reward
            done = terminated or truncated
            if(info['x_pos'] >= 1200):
                print("stop")
            if done:
                if(metrics): runtime = (time.time_ns() - start) / 1000000000
                if(self.DEBUG is not None): print(f"Reward for run: {run_score}")
                break
            step += 1
        
        self.env.close() # Uncomment to end run after death
        if(metrics and done):
            data = {
                'run-score': run_score,
                'run-time': runtime,
                'steps': step,
                'x_pos': info['x_pos']
            }
            return data
        else: return


if(__name__ == "__main__"):
    print("Please use run_agent.py to run the agent")