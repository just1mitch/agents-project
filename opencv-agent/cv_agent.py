import mario_locate_objects
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np




# Main game playing agent
class CVAgent:
    # A class holding information of the last state of the game
    # Allows heuristics to have a limited understanding of
    # the current occurrences of the game
    class LastState:
        last_action = None
        mario_world_coords = None
        mario_status = None
        enemy_locations = ()
        block_locations = ()
        item_locations  = ()
    
    last_state = LastState()
    env = None
    STEPS_PER_ACTION = 5
    GOOMBA_RANGE = 55
    KOOPA_RANGE = 40
    jumping_hole = False # State of mario if he is jumping a hole
    jumping_enemy = False

    def __init__(self):
        self.env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="human")
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
                    return True
            # Found no holes, return empty list
            return False
        return False

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
                            if(enemy[0][0] - (mario_locations[0][0][0] + mario_locations[0][1][0]) in range(jump_range)):
                                return True

    # Make action function adapted from Lauren Gee's work
    # In mario_locate_objects.py
    def __make_action(self, screen, info, step, prev_action):
        
        game_step = step % self.STEPS_PER_ACTION
        if game_step != 0 and game_step != self.STEPS_PER_ACTION - 3: 
            return prev_action
        # 2 steps before an action is chosen, a snapshot of the game
        # is saved in a LastState object to be referenced when making
        # a choice on the next move
        elif(game_step == self.STEPS_PER_ACTION - 3):
            self.last_state.last_action = prev_action
            self.last_state.mario_world_coords = (info["x_pos"], info["y_pos"])
            self.last_state.mario_status = info["status"]

            # This is the format of the lists of locations:
            # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
            #
            # x_coordinate and y_coordinate are the top left corner of the object
            #
            # For example, the enemy_locations list might look like this:
            # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
            object_locations = mario_locate_objects.locate_objects(screen, self.last_state.mario_status)
            self.last_state.enemy_locations = object_locations["enemy"]
            self.last_state.block_locations = object_locations["block"]
            self.last_state.item_locations  = object_locations["item"]

            # Grabbed the info to store for next action, now return prev_action
            return prev_action
        
        elif(game_step == 0):
            mario_status = info["status"]
            object_locations = mario_locate_objects.locate_objects(screen, mario_status)

            mario_locations = object_locations["mario"]
            enemy_locations = object_locations["enemy"]
            block_locations = object_locations["block"]
            item_locations = object_locations["item"]

            if(len(mario_locations) != 1):
                # Mario cannot be found, return action 1 for now
                action = 1
                return action
            
            # state = self.__get_mario_state(mario_locations[0], screen)
            # Secondary check for mario being on ground, as the sprites can be inconsistent
            block_under_mario = self.__block_under_mario(block_locations, mario_locations[0])
            if(block_under_mario != None): state = 'G'
            else: state = 'A'

            if(prev_action in [2,5] and state == 'G'): return 1 # run right to allow jump again

            # Mario displacement based on previous frames
            mario_disp = (info["x_pos"] - self.last_state.mario_world_coords[0], info["y_pos"] - self.last_state.mario_world_coords[1])
            # If mario on ground, follow decision tree
            if (state == 'G'):
                if(self.jumping_hole): self.jumping_hole = False # no longer jumping if on ground again
                if(self.jumping_enemy): self.jumping_enemy = False

                enemy_found = self.__check_enemies(enemy_locations, mario_locations, mario_disp)
                if(enemy_found):
                    action = 2
                    self.jumping_enemy = True
                    return action

                for block in block_locations:
                    # Check for pipes
                    if(block[2] == 'pipe'):
                        pipe_range = (10, 60) # How close (in pixels) mario should get before jumping over pipe - lowerbound, upperbound
                        mario_to_pipe = block[0][0] - mario_locations[0][0][0]
                        if(mario_to_pipe in range(pipe_range[0], pipe_range[1])):
                            # note - by doing (block - mario) we eliminate pipes that are to the left of mario
                            action = 2
                            return action
                    # Check if need to jump over a block
                    else:
                        block_range = (10, 30) # How close mario should get before jumping over block in path
                        if(block[0][0] - mario_locations[0][0][0] in range(block_range[0], block_range[1])):
                            # if halfway through mario cuts through block
                            if(mario_locations[0][0][1] + (mario_locations[0][1][1] / 2) in range(block[0][1], block[0][1] + block[1][1])):
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
                                    action = 5
                                    return action
  
            action = 1
            return action


    def play(self, debug=False):
        obs = None
        done = True
        self.env.reset()
        step = 0
        run_score = 0
        if(debug): print("Running in debug mode")
        while True:
            if obs is not None:
                action = self.__make_action(obs, info, step, action)
            else:
                action = 1
            obs, reward, terminated, truncated, info = self.env.step(action)
            run_score += reward
            done = terminated or truncated
            if done:
                if(debug): print(f"Reward for run: {run_score}")
                step = 0
                run_score = 0
                self.env.reset()
            step += 1
        self.env.close()