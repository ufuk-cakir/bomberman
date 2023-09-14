
import torch
import os
import pickle
import random

from .Qmodel import QNet, Memory, Transition, HYPER, ACTIONS
from collections import namedtuple, deque
import torch

import numpy as np
import math

#----------------------------------

from items import Bomb


import numpy as np

import settings

VIEW_SIZE: int = 7

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

def get_object_map(object_xy_list):
    object_map = np.zeros((settings.COLS, settings.ROWS))
    for (x, y) in object_xy_list:
        object_map[x, y] = 1
    return object_map


def view_port_state(game_state: dict) -> np.ndarray:
    """
    Features per field:

     - 0: Free
     - 1: Breakable
     - 2: Obstructed
     - 3: Contains player
     - 4: Contains coin
     - 5: Danger level
     - 6: Contains explosion
     !- 7: Contains opponent
    """
    num_features_per_tile = 7

    feature_shape: tuple = (VIEW_SIZE * VIEW_SIZE * num_features_per_tile,)
    features = np.full(feature_shape, np.nan)
    _, _, _, (player_x, player_y) = game_state["self"]
    coins = game_state["coins"]
    opponent_coords = [(x, y) for _, _, _, (x, y) in game_state["others"]]

    coin_map = get_object_map(coins)
    opponent_map = get_object_map(opponent_coords)

    origin_x = player_x
    origin_y = player_y
    if (origin_x - VIEW_SIZE // 2) < 0:
        origin_x = VIEW_SIZE // 2
    if (origin_y - VIEW_SIZE // 2) < 0:
        origin_y = VIEW_SIZE // 2

    if (origin_x + VIEW_SIZE // 2) >= settings.COLS:
        origin_x = settings.COLS - VIEW_SIZE // 2 - 1
    if (origin_y + VIEW_SIZE // 2) >= settings.ROWS:
        origin_y = settings.ROWS - VIEW_SIZE // 2 - 1

    x_range = range(origin_x - VIEW_SIZE // 2, origin_x + VIEW_SIZE // 2 + 1)
    y_range = range(origin_y - VIEW_SIZE // 2, origin_y + VIEW_SIZE // 2 + 1)
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            field_index = (
                np.ravel_multi_index((i, j), (VIEW_SIZE, VIEW_SIZE))
                * num_features_per_tile
            )
            field = game_state["field"][x, y]
            features[field_index + 0] = int(field == 0) - 0.5
            features[field_index + 1] = int(field == 1) - 0.5
            features[field_index + 2] = int(np.abs(field) == 1) - 0.5
            features[field_index + 3] = int(player_x == x and player_y == y) - 0.5
            features[field_index + 4] = coin_map[x, y] - 0.5
            features[field_index + 5] = 0
            for (bomb_x, bomb_y), timer in game_state["bombs"]:
                if bomb_x == x and bomb_y == y:
                    features[field_index + 5] = (
                        settings.BOMB_TIMER - timer
                    ) / settings.BOMB_TIMER
                    break
            features[field_index + 6] = int(opponent_map[x, y]) - 0.5
    assert np.all(~np.isnan(features))
    return features

#-------------------------





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

    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.steps_done = 0
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    if self.train or not os.path.isfile("policy_net.pt"):
        if CONTINUE_TRAINING:
            self.logger.info("Loading model from saved state.")
            with open("policy_net.pt", "rb") as file:
                self.policy_net = pickle.load(file)
            with open("target_net.pt", "rb") as file:
                self.target_net = pickle.load(file)
            return
        else:
            self.logger.info("Setting up model from scratch.")
            self.logger.info(f"Number of possible actions: {len(ACTIONS)}"
                             f"Size of state vector: {HYPER.N_FEATURES}")
            weights = np.random.rand(len(ACTIONS))
            #self.model = weights / weights.sum()
            self.policy_net = QNet(HYPER.N_FEATURES, HYPER.N_ACTIONS).to(self.device)
            self.target_net = QNet(HYPER.N_FEATURES, HYPER.N_ACTIONS).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
    else:
        self.logger.info("Loading model from saved state.")
        with open("policy_net.pt", "rb") as file:
            self.policy_net = pickle.load(file)
        with open("target_net.pt", "rb") as file:
            self.target_net = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Get State feature vector
    self.policy_net.eval()
    s = state_to_features(self,game_state)
    s = torch.tensor(s, device=self.device, dtype=torch.float).unsqueeze(0)
     
    # Epsilon greedy policy
    sample = random.random() # sample from uniform distribution
    if not self.train:
        eps_threshold = 0
    else: 
        eps_threshold = HYPER.EPS_END + (HYPER.EPS_START - HYPER.EPS_END) * \
        math.exp(-1. * self.steps_done / HYPER.EPS_DECAY)
    self.steps_done += 1
    # If sample is smaller than epsilon, choose random action
    if sample < eps_threshold:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)
    else:
        self.logger.debug("Querying model for action.")
        # Choose action with highest Q value
        with torch.no_grad():
            
            action = self.policy_net(s).max(1)[1].view(1, 1).item()
            return ACTIONS[action]

# Copied from rule_based agent TODO
import numpy as np
from random import shuffle
def look_for_targets(free_space, start, targets,logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """

    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

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

    # 1. Distance to Nearest Coin
    distances_to_coins = [np.abs(x-cx) + np.abs(y-cy) for (cx, cy) in coins]
    nearest_coin_distance = min(distances_to_coins) if coins else -1


    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    
    
    # 2.Dead End
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    directions = [(x+dx, y+dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    free_spaces = sum(1 for d in directions if arena[d] == 0)
    
    is_dead_end = 1 if free_spaces == 1 else 0 #TODO maybe remove this

    # 3. Nearby Bomb Threat and 4. Bomb's Time to Explosion
    '''
    bomb_threat = 0
    time_to_explode = 5  # Assuming max time is 4 for a bomb to explode TODO is this correct?
    for (bx, by), t in bombs:
        if abs(bx - x) < 4 or abs(by - y) < 4:
            bomb_threat = 1
            time_to_explode = min(time_to_explode, t)
    '''
    
    # Check blast range of each bomb
    blast_coords = []
    
    for (bx, by), t in bombs:
        # Start with the bomb position itself
        blast_coords.append((bx, by))
        # For each direction, check for crates
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: # TODO maybe add diagonal directions
            for i in range(1, 4):
                # Stop when hitting a wall
                if arena[bx + i*dx, by + i*dy] == -1:
                    break
                # Add free spaces
                if arena[bx + i*dx, by + i*dy] == 0:
                    blast_coords.append((bx + i*dx, by + i*dy))
                # Stop when hitting a crate
                if arena[bx + i*dx, by + i*dy] == 1:
                    blast_coords.append((bx + i*dx, by + i*dy))
                    break
    

                   
    # Check if agent is in bomb blast range
    bomb_threat = 1 if (x, y) in blast_coords else 0
    # additionaly check if agent is in explosion map
    if explosion_map[x, y] > 0:
        bomb_threat = 1


   
    # 5. Can Drop Bomb
    can_drop_bomb = 1 if bombs_left > 0 and (x, y) not in bomb_xys else 0

    # 6. Is Next to Opponent
    is_next_to_opponent = 1 if any(abs(ox-x) + abs(oy-y) == 1 for (ox, oy) in others) else 0

    # 7. Is on Bomb
    is_on_bomb = 1 if (x, y) in bomb_xys else 0

    # 8. Number of targets
    #crates_nearby = sum(1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)] if arena[x+dx, y+dy] == 1)
    
    # 9. Escape Route Available (simplified for brevity)
    escape_route_available = 1 if free_spaces > 1 else 0 #TODO implement this correctly

    # 10. Direction to Nearest Target (simplified for brevity)
    # Assuming the function look_for_targets returns a direction as (dx, dy)
    targets = coins + dead_ends + crates + others # TODO add others? ADD flag if others are importatnt
    # Exclude targets that are occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    
    # Exclude targets that are currently in explosion map
    targets = [targets[i] for i in range(len(targets)) if explosion_map[targets[i]] == 0]

    # Exlucude coordinates that are in blast range of a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in blast_coords]
    
    target_direction = look_for_targets(arena == 0, (x, y), targets)

    direction_to_target = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    if target_direction:
        if target_direction == (x, y-1): direction_to_target[0] = 1
        elif target_direction == (x, y+1): direction_to_target[1] = 1
        elif target_direction == (x-1, y): direction_to_target[2] = 1
        elif target_direction == (x+1, y): direction_to_target[3] = 1

    
    if (x, y) in dead_ends:
        should_drop_bomb = 1
    if (x, y) in dead_ends:
        should_drop_bomb = 1
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            should_drop_bomb = 1
    # Add proposal to drop a bomb if arrived at target and touching crate
    if target_direction == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        should_drop_bomb = 1
    else:
        should_drop_bomb = 0 # TODO check if this is correct, maybe other situations to drop bomb
    # 11. Is in a Loop 
    is_in_loop = 1 if self.coordinate_history.count((x, y)) > 2 else 0
    self.coordinate_history.append((x, y))
    
    # 12. Ignore Others Timer (normalized)
    #ignore_others_timer_normalized = self.ignore_others_timer / 5  # Assuming max timer is 5
 
        
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
    
    # Combining all features into a single list
    features = [
        nearest_coin_distance, is_dead_end, bomb_threat, time_to_explode,
        can_drop_bomb, is_next_to_opponent, is_on_bomb, should_drop_bomb,
        escape_route_available
    ] + direction_to_target + [is_in_loop, nearest_bomb_distance] + blast_in_direction + danger_level #ignore_others_timer_normalized]


    return np.array(features)