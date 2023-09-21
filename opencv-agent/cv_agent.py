import mario_locate_objects
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym




# Main game playing agent
class CVAgent:
    # A class holding information of the last state of the game
    # Allows heuristics to have a limited understanding of
    # the current occurrences of the game
    class LastState:
        last_action = None
        mario_status = None
        mario_locations = ()
        enemy_locations = ()
        block_locations = ()
        item_locations  = ()
    
    last_state = LastState()
    env = None

    def __init__(self):
        self.env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
        self.env = gym.wrappers.GrayScaleObservation(self.env)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
    
    # Make action function adapted from Lauren Gee's work
    # In mario_locate_objects.py
    def __make_action(self, screen, info, step, prev_action):
        
        game_step = step % 10
        if game_step != 0 and game_step != 8: 
            return prev_action
        # 2 steps before an action is chosen, a snapshot of the game
        # is saved in a LastState object to be referenced when making
        # a choice on the next move
        elif(game_step == 8):
            self.last_state.last_action = prev_action
            self.last_state.mario_status = info["status"]
            
            # This is the format of the lists of locations:
            # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
            #
            # x_coordinate and y_coordinate are the top left corner of the object
            #
            # For example, the enemy_locations list might look like this:
            # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
            object_locations = mario_locate_objects.locate_objects(screen, self.last_state.mario_status)
            self.last_state.mario_locations = object_locations["mario"]
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
            action = 2
            #self.env.action_space.sample()
            return action


    def play(self):
        obs = None
        done = True
        self.env.reset()
        for step in range(5000):
            if obs is not None:
                action = self.__make_action(obs, info, step, action)
            else:
                action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if done:
                self.env.reset()
        self.env.close()