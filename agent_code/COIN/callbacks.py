

import os
import pickle
import random

from .ppo import PPO,HYPER
import torch
import math

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']






GET_INPUT = False

if GET_INPUT:

    CONTINUE_TRAINING = None


# Ask from terminal wheter to continue training or not
    CONTINUE_TRAINING = input("Continue training? (y/n)")
    if CONTINUE_TRAINING == "y":
        CONTINUE_TRAINING = True

    else:
        CONTINUE_TRAINING = False
        
    LOG_WANDB = input("Log to wandb? (y/n)")
    if LOG_WANDB == "y":
        LOG_WANDB = True
    else:
        LOG_WANDB = False
        
    debug_events = input("Debug events while training? (y/n)")
    if debug_events == "y":
        DEBUG_EVENTS = True
    else:
        DEBUG_EVENTS = False
        
    log_to_file = input("Log to file? (y/n)")
    if log_to_file == "y":
        LOG_TO_FILE = True
    else:
        LOG_TO_FILE = False


    log_features = input("Log features? (y/n)")
    if log_features == "y":
        log_features = True
    else:
        log_features = False


else:
    CONTINUE_TRAINING = True
    LOG_WANDB = True
    DEBUG_EVENTS = False
    LOG_TO_FILE = False
    log_features = False
    


from collections import deque

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.steps_done = 0
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",self.device)
    self.logger.info("CONTINUE_TRAINING: " + str(CONTINUE_TRAINING))
    self.logger.info("LOG_WANDB: " + str(LOG_WANDB))
    self.logger.info("DEBUG_EVENTS: " + str(DEBUG_EVENTS))
    self.logger.info("LOG_TO_FILE: " + str(LOG_TO_FILE))
    self.logger.info("log_features: " + str(log_features))
    self.logger.info("device: " + str(self.device))
    if self.train or not os.path.isfile(HYPER.MODEL_NAME):
        if CONTINUE_TRAINING:
            self.logger.info("Loading model from saved state.")
            self.model = PPO(NUMBER_OF_POSSIBLE_ACTIONS=len(ACTIONS), SIZE_OF_STATE_VECTOR=HYPER.SIZE_OF_STATE_VECTOR)
            self.model.load_state_dict(torch.load(HYPER.MODEL_NAME))
            return
        else:
            self.logger.info("Setting up model from scratch.")
            self.logger.info(f"Number of possible actions: {len(ACTIONS)}"
                             f"Size of state vector: {HYPER.SIZE_OF_STATE_VECTOR}")
            weights = np.random.rand(len(ACTIONS))
            #self.model = weights / weights.sum()
            self.model = PPO(NUMBER_OF_POSSIBLE_ACTIONS=len(ACTIONS), SIZE_OF_STATE_VECTOR=HYPER.SIZE_OF_STATE_VECTOR)
    else:
        self.logger.info("Loading model from saved state. No training will be performed.")
        with open(HYPER.MODEL_NAME, "rb") as file:
            #self.model = pickle.load(file)
            self.model = PPO(NUMBER_OF_POSSIBLE_ACTIONS=len(ACTIONS), SIZE_OF_STATE_VECTOR=HYPER.SIZE_OF_STATE_VECTOR)
            self.model.load_state_dict(torch.load(HYPER.MODEL_NAME, map_location=self.device))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # Get State feature vector
    s = state_to_features(self,game_state)
   
    # Epsilon greedy policy TODO implement annealing, improve
    sample = random.random() # sample from uniform distribution
    eps_threshold = HYPER.EPS_END + (HYPER.EPS_START - HYPER.EPS_END) * \
        math.exp(-1. * self.steps_done / HYPER.EPS_DECAY)
    self.steps_done += 1
    
    if not self.train:
        # Reduce exploration when not training
        eps_threshold = 0.00
        
    # If sample is smaller than epsilon, choose random action
    if sample < eps_threshold:
       # self.logger.debug("Choosing action purely at random.")
        self.prob_a = 1/len(ACTIONS)
        self.RANDOM_ACTION = True
        return np.random.choice(ACTIONS)
    # Get action probabilities
    prob_distr = self.model.pi(torch.from_numpy(s).float().to(self.device))
    categorical = torch.distributions.Categorical(prob_distr)
    
    # Sample action
    a = categorical.sample().item()
    action = ACTIONS[a]
    if log_features:
        self.logger.debug(f"Action: {action}")
        self.logger.debug(f"Probabilities: {prob_distr}")
    self.prob_a = prob_distr[a].item()
    self.RANDOM_ACTION = False
    return action 


def get_danger_level(self,agent_position, explosion_map, arena, debug = True):
    x, y = agent_position
    danger_level = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    if debug: self.logger.debug(f'Agent position: {(x, y)}')
    for i, (dx, dy) in enumerate(directions):
        tile_x, tile_y = x + dx, y + dy
        if debug: self.logger.debug(f'Checking tile {(tile_x, tile_y)}')
        if debug: self.logger.debug(f'Explosion map value: {explosion_map[tile_x, tile_y]}')
        # Check if the tile is within the arena boundaries
        if 0 <= tile_x < arena.shape[0] and 0 <= tile_y < arena.shape[1]:
            danger_level[i] = int(explosion_map[tile_x, tile_y])
            self.logger.debug(f'Danger level: {danger_level}')
    
    return danger_level



def state_to_features(self,game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it end
    # Extracting basic information
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    explosion_map = game_state['explosion_map'] 
    
    explosion_coords = [(x,y) for x in range(arena.shape[0]) for y in range(arena.shape[1]) if explosion_map[x,y] > 0]
   
    #Distance to Nearest Coin
    distances_to_coins = [np.abs(x-cx) + np.abs(y-cy) for (cx, cy) in coins]
    nearest_coin_distance = min(distances_to_coins) if coins else -1


    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
      
    # Check blast range of each bomb
    blast_coords = []
    blast_coords_timer = []
    
    for (bx, by), t in bombs:
        # Start with the bomb position itself
        blast_coords.append((bx, by))
        blast_coords_timer.append(t)
        # For each direction, check for crates
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: # TODO maybe add diagonal directions
            for i in range(1, 4):
                # Stop when hitting a wall
                if arena[bx + i*dx, by + i*dy] == -1:
                    break
                # Add free spaces
                if arena[bx + i*dx, by + i*dy] == 0:
                    blast_coords.append((bx + i*dx, by + i*dy))
                    blast_coords_timer.append(t)
                # Stop when hitting a crate
                if arena[bx + i*dx, by + i*dy] == 1:
                    blast_coords.append((bx + i*dx, by + i*dy))
                    blast_coords_timer.append(t)
                    break
    

                   
    # Check if agent is in bomb blast range
    bomb_threat = 1 if (x, y) in blast_coords else 0
    # additionaly check if agent is in explosion map
    if explosion_map[x, y] > 0:
        bomb_threat = 1


    # Can Drop Bomb
    can_drop_bomb = 1 if bombs_left > 0 and (x, y) not in bomb_xys else 0

    # Is Next to Opponent
    is_next_to_opponent = 1 if any(abs(ox-x) + abs(oy-y) == 1 for (ox, oy) in others) else 0

    # Is on Bomb
    is_on_bomb = 1 if (x, y) in bomb_xys else 0

    # Is in a Loop 
    is_in_loop = 1 if self.coordinate_history.count((x, y)) > 3 else 0
    self.coordinate_history.append((x, y))

     
    blast_in_direction = [0, 0, 0, 0] # [UP, DOWN, LEFT, RIGHT]    
    if bomb_threat:
        # Count number of tiles occupied by blast in each direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    
        # Iterate through each direction
        for i, (dx, dy) in enumerate(directions):
            for j in range(1, 4):  # Check up to 3 tiles in each direction
                # Calculate the coordinates of the tile in the current direction
                tile_x, tile_y = x + j*dx, y + j*dy
                # Check if the tile is in the blast range
                if (tile_x, tile_y) in blast_coords:
                    blast_in_direction[i] += 1
                if (tile_x,tile_y) in explosion_coords:
                   
                    danger_level[i] += 1
                # If the tile is a wall, stop checking further in this direction
                elif arena[tile_x, tile_y] == -1:
                    break
                
    danger_level = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    if len(explosion_coords) > 0:
        # Count number of tiles occupied by blast in each direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    
        # Iterate through each direction
        for i, (dx, dy) in enumerate(directions):
            for j in range(1, 4):  # Check up to 3 tiles in each direction
                # Calculate the coordinates of the tile in the current direction
                tile_x, tile_y = x + j*dx, y + j*dy
             
                if (tile_x,tile_y) in explosion_coords:
                    
                    danger_level[i] += 1
                # If the tile is a wall, stop checking further in this direction
                elif arena[tile_x, tile_y] == -1:
                    break
            

        
    

    # Distance to closest bomb
    distance_to_bombs = [np.abs(x-bx) + np.abs(y-by) for (bx, by) in bomb_xys]
    nearest_bomb_distance = min(distance_to_bombs) if bomb_xys else -1
    
    # time to explode
    time_to_explode = 5  # Assuming max time is 4 for a bomb to explode TODO is this correct?
    for (bx, by), t in bombs:
        if abs(bx - x) < 4 or abs(by - y) < 4:
           time_to_explode = min(time_to_explode, t)

    # Get angle to nearest coin
    if len(coins) > 0:
        nearest_coin = coins[np.argmin(distances_to_coins)]
        angle_to_nearest_coin = np.arctan2(nearest_coin[1] - y, nearest_coin[0] - x)
       
    else:
        angle_to_nearest_coin = -1
    
    
    # Add direction to nearest coin: 
    # Encode if the coin is above or below the agent in first bit, and if it is left or right in second bit
    direction_to_coin = [-1, -1]
    if len(coins) > 0:
        nearest_coin = coins[np.argmin(distances_to_coins)]
        if nearest_coin[0] - x > 0:
            direction_to_coin[1] = 1 # Coin is to the right
        else:
            direction_to_coin[1] = 0 # Coin is to the left
            
        if nearest_coin[1] - y > 0:
            direction_to_coin[0] = 1 # Coin is below
        else:
            direction_to_coin[0] = 0 # Coin is above
            
    
    # local map 
    local = np.zeros((5,5)) -2.5
    
    walls = [(xs,ys) for xs in range(arena.shape[0]) for ys in range(arena.shape[1]) if arena[xs,ys] == -1]
    
    # get the coordinates of the local map view
    local_coords = [(xs,ys) for xs in range(x-2,x+3) for ys in range(y-2,y+3)]

    # fill in arena values
    for (xs,ys) in local_coords:
        if (xs,ys) in walls:
            local[xs-x+2,ys-y+2] = -2.5
        elif (xs,ys) in coins:
            local[xs-x+2,ys-y+2] = 1.5
        elif (xs,ys) in crates:
            local[xs-x+2,ys-y+2] = 1
        elif (xs,ys) in others:
            local[xs-x+2,ys-y+2] = -4.5
        elif (xs,ys) in explosion_coords:
            local[xs-x+2,ys-y+2] = -explosion_map[xs,ys]-0.125
        elif (xs,ys) in blast_coords:
            # Fill in with the timer of bomb explosion
            index = blast_coords.index((xs,ys))
            local[xs-x+2,ys-y+2] = -blast_coords_timer[index]-0.125
        elif (xs,ys) == (x,y):
            local[xs-x+2,ys-y+2] = 5
        elif xs < 0 or xs > 16 or ys < 0 or ys > 16:
            local[xs-x+2,ys-y+2] = -2.5
        
        else:
            local[xs-x+2,ys-y+2] = 0
     
    
    # Start with agent position
    local[2,2] = 5 # agent position
    local = local.flatten().tolist()
    

    valid_directions = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    for i, (dx, dy) in enumerate(directions):
        tile_x, tile_y = x + dx, y + dy
        # check if the tile is free
        if arena[tile_x, tile_y] ==0:
            valid_directions[i] = 1
    
    
    features = valid_directions+[
        nearest_coin_distance, angle_to_nearest_coin] + direction_to_coin + [ bomb_threat, time_to_explode,
        can_drop_bomb, is_next_to_opponent, is_on_bomb, 
    ] + [is_in_loop, nearest_bomb_distance] + blast_in_direction + danger_level + local #ignore_others_timer_normalized]


    if log_features:
        self.logger.info(f"Nearest Coin Distance: {nearest_coin_distance}")
        self.logger.info(f"Angle to Nearest Coin: {angle_to_nearest_coin}")
        self.logger.info(f"Direction to Coin: {direction_to_coin}")
        self.logger.info(f"Bomb Threat: {bomb_threat}")
        self.logger.info(f"Time to Explode: {time_to_explode}")
        self.logger.info(f"Can Drop Bomb: {can_drop_bomb}")
        self.logger.info(f"Is Next to Opponent: {is_next_to_opponent}")
        self.logger.info(f"Is on Bomb: {is_on_bomb}")
        self.logger.info(f"Is in Loop: {is_in_loop}")
        self.logger.info(f"Nearest Bomb Distance: {nearest_bomb_distance}")
        self.logger.info(f"Blast in Direction: {blast_in_direction}")
        self.logger.info(f'Danger level: {danger_level}')
        self.logger.info(f"Local Map:\n {np.array(local).reshape((5,5)).T}")
    return(np.array(features))
