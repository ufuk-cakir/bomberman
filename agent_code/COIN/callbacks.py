

import os
import pickle
import random

from .ppo import PPO,HYPER
import torch
import math

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



CONTINUE_TRAINING = 1

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
    if self.train or not os.path.isfile(HYPER.MODEL_NAME):
        if CONTINUE_TRAINING:
            self.logger.info("Loading model from saved state.")
            with open(HYPER.MODEL_NAME, "rb") as file:
                self.model = pickle.load(file)
            return
        else:
            self.logger.info("Setting up model from scratch.")
            weights = np.random.rand(len(ACTIONS))
            #self.model = weights / weights.sum()
            self.model = PPO(NUMBER_OF_POSSIBLE_ACTIONS=len(ACTIONS), SIZE_OF_STATE_VECTOR=HYPER.SIZE_OF_STATE_VECTOR)
    else:
        self.logger.info("Loading model from saved state.")
        with open(HYPER.MODEL_NAME, "rb") as file:
            self.model = pickle.load(file)


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
        #self.logger.debug("Choosing action purely at random.")
        self.prob_a = 1/len(ACTIONS)
        return np.random.choice(ACTIONS)
    # Get action probabilities
    prob_distr = self.model.pi(torch.from_numpy(s).float())
    categorical = torch.distributions.Categorical(prob_distr)
    
    # Sample action
    a = categorical.sample().item()
    action = ACTIONS[a]
    self.prob_a = prob_distr[a].item()
    return action 

    
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)

#from .feature_selection_new import create_features
# is imported from feature_selection.py


# Copied from rule_based agent TODO
import numpy as np
from random import shuffle
def look_for_targets(free_space, start, targets, logger=None):
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






# TODO implment ignore others timer like in rule_based_agent
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

    # 1. Distance to Nearest Coin
    distances_to_coins = [np.abs(x-cx) + np.abs(y-cy) for (cx, cy) in coins]
    nearest_coin_distance = min(distances_to_coins) if coins else -1

    # 2. Is Dead End
    directions = [(x+dx, y+dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    free_spaces = sum(1 for d in directions if arena[d] == 0)
    is_dead_end = 1 if free_spaces == 1 else 0

    # 3. Nearby Bomb Threat and 4. Bomb's Time to Explosion
    bomb_threat = 0
    time_to_explode = 5  # Assuming max time is 4 for a bomb to explode TODO is this correct?
    for (bx, by), t in bombs:
        if abs(bx - x) < 4 or abs(by - y) < 4:
            bomb_threat = 1
            time_to_explode = min(time_to_explode, t)

    # 5. Can Drop Bomb
    can_drop_bomb = 1 if bombs_left > 0 and (x, y) not in bomb_xys else 0

    # 6. Is Next to Opponent
    is_next_to_opponent = 1 if any(abs(ox-x) + abs(oy-y) == 1 for (ox, oy) in others) else 0

    # 7. Is on Bomb
    is_on_bomb = 1 if (x, y) in bomb_xys else 0

    # 8. Number of Crates Nearby
    crates_nearby = sum(1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)] if arena[x+dx, y+dy] == 1)

    # 9. Escape Route Available (simplified for brevity)
    escape_route_available = 1 if free_spaces > 1 else 0

    # 10. Direction to Nearest Target (simplified for brevity)
    # Assuming the function look_for_targets returns a direction as (dx, dy)
    target_direction = look_for_targets(arena == 0, (x, y), coins + others)
    direction_to_target = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    if target_direction:
        if target_direction == (x, y-1): direction_to_target[0] = 1
        elif target_direction == (x, y+1): direction_to_target[1] = 1
        elif target_direction == (x-1, y): direction_to_target[2] = 1
        elif target_direction == (x+1, y): direction_to_target[3] = 1

    # 11. Is in a Loop 
    is_in_loop = 1 if self.coordinate_history.count((x, y)) > 2 else 0
    self.coordinate_history.append((x, y))
    
    # 12. Ignore Others Timer (normalized)
    #ignore_others_timer_normalized = self.ignore_others_timer / 5  # Assuming max timer is 5

    # Combining all features into a single list
    features = [
        nearest_coin_distance, is_dead_end, bomb_threat, time_to_explode,
        can_drop_bomb, is_next_to_opponent, is_on_bomb, crates_nearby,
        escape_route_available
    ] + direction_to_target + [is_in_loop] #ignore_others_timer_normalized]
    
    # Add a local view of the map as a feature, hyperparameter LOCAL_VIEW_SIZE
    
    #local_map = arena[x-HYPER.LOCAL_VIEW_SIZE:x+HYPER.LOCAL_VIEW_SIZE+1,y-HYPER.LOCAL_VIEW_SIZE:y+HYPER.LOCAL_VIEW_SIZE+1]
    #local_map = local_map.flatten() # flatten to 1D array
    
   # features = features + local_map.tolist()
   # self.logger.debug(f'Raw features: {features}')
    # Normalizing featur
    
    
    #self.logger.debug("features: " + str(features))
    return np.array(features)