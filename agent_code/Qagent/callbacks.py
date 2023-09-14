
import torch
import os
import pickle
import random

from .Qmodel import QNet, Memory, Transition, HYPER, ACTIONS
from collections import namedtuple
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
    
    
    
    
    if not CONTINUE_TRAINING or not os.path.isfile("policy_net.pt"):
        self.logger.info("Setting up model from scratch.")
        self.policy_net = QNet(HYPER.N_FEATURES, HYPER.N_ACTIONS).to(self.device)
        self.target_net = QNet(HYPER.N_FEATURES, HYPER.N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy weights and stuffweights = np.random.rand(len(ACTIONS))
       # Optimizer etc setup in setup_training in train.py
        
        
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


    """
def state_to_features(self,game_state: dict) -> np.array:
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Get self information from game state
    _, self_score,self_bomb_possible, (self_x, self_y) = game_state['self']
    # Get board information from game state
    board = np.array(game_state['field'])
    coins = np.array(game_state['coins'])
    
    # There can be a maximum of 4 coins at the same, time so we need to pad the array

    if len(coins)==0:
        coins = np.array([[-1,-1],[-1,-1],[-1,-1],[-1,-1]])
    if len(coins) < 4 and len(coins)>0:
        for i in range(4 - len(coins)):
            coins = np.append(coins, [[-1, -1]], axis=0)
    # Get bomb information from game state
    bombs = game_state['bombs']
    bomb_xy = np.array([xy for (xy,t) in bombs])

    # There can be a maximum of 4 bombs at the same, time so we need to pad the array
    if len(bomb_xy)==0:
        bomb_xy = np.array([[-1,-1],[-1,-1],[-1,-1],[-1,-1]])
    if len(bomb_xy) < 4 and len(bomb_xy)>0:
        for i in range(4 - len(bomb_xy)):
            bomb_xy = np.append(bomb_xy, [[-1, -1]], axis=0)

    others_xy = np.array([[-1,-1],[-1,-1],[-1,-1],[-1,-1]])
    others = np.array([xy for (name, score, bomb_possible, xy) in game_state['others']])
    others_xy[:len(others)] = others
        # Create a list of arrays
    channel_arrays = [
        board.flatten(),
        np.array([self_x]),
        np.array([self_y]),
        coins.flatten(),
        bomb_xy.flatten(),
        others_xy.flatten(),
        np.array([self_bomb_possible])
    ]

    # Concatenate the arrays along axis 1 to create the channels array
    channels = np.concatenate(channel_arrays, axis=0)
    # Return flattened feature tensor to feed into the model
    return channels# For example, you could construct several channels of equal shape, ...



def calculate_valid_directions(self, self_x,self_y,board) -> list:
    '''Calcualtes the valid actions for the current state of the game.'''
    # Check if the agent can move in the four directions
    valid_actions = [0,0,0,0] # [up, down, left, right]
    if self_y > 0 and board[self_y - 1, self_x] == 0:
        valid_actions[0] = 1
    if self_y < board.shape[0] - 1 and board[self_y + 1, self_x] == 0:
        valid_actions[1] = 1
    if self_x > 0 and board[self_y, self_x - 1] == 0:
        valid_actions[2] = 1
    if self_x < board.shape[1] - 1 and board[self_y, self_x + 1] == 0:
        valid_actions[3] = 1
    return valid_actions


def find_nearest_coin(self, self_x, self_y, coins) -> int:
    '''Finds the nearest coin to the agent.'''
    # Calculate the distance to each coin
    distances = np.sum(np.abs(coins - np.array([self_x, self_y])), axis=1)
    # Return the index of the nearest coin
    return np.argmin(distances)
            

def state_to_features(self, game_state: dict) -> torch.Tensor:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Get self information from game state
    _, self_score, self_bomb_possible, (self_x, self_y) = game_state['self']
    # Get board information from game state
    board = torch.tensor(game_state['field'], dtype=torch.float32)
    coins = torch.tensor(game_state['coins'], dtype=torch.float32)
    
    
    
    # Pad the coins tensor if necessary
    if coins.size(0) < 4:
        pad_size = 4 - coins.size(0)
        padding = torch.zeros((pad_size, 2), dtype=torch.float32)
        coins = torch.cat([coins, padding])

    # Get bomb information from game state
    bombs = game_state['bombs']
    bomb_xy = torch.tensor([xy for (xy, t) in bombs], dtype=torch.float32)
    
    # Pad the bomb_xy tensor if necessary
    if bomb_xy.size(0) < 4:
        pad_size = 4 - bomb_xy.size(0)
        padding = torch.zeros((pad_size, 2), dtype=torch.float32)
        bomb_xy = torch.cat([bomb_xy, padding])

    others_xy = torch.zeros((4, 2), dtype=torch.float32)
    others = torch.tensor([(xy[0], xy[1]) for (_, _, _, xy) in game_state['others']], dtype=torch.float32)
    if others.size(0) > 0:
        others_xy[:others.size(0)] = others
    
    # Create a list of tensors
    channel_tensors = [
        board.view(-1),
        torch.tensor([self_x], dtype=torch.float32),
        torch.tensor([self_y], dtype=torch.float32),
        coins.view(-1),
        bomb_xy.view(-1),
        others_xy.view(-1),
        torch.tensor([self_bomb_possible], dtype=torch.float32)
    ]

    # Concatenate the tensors along axis 0 to create the channels tensor
    channels = torch.cat(channel_tensors)
    # Return the feature tensor to feed into the model
    return channels

    """



explosion_escape_combinations = [
    [[1, 0]],
    [[1, 0], [1, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, 0], [2, 1]],
    [[1, 0], [2, 0], [3, 0]],
    [[1, 0], [2, 0], [3, 0], [3, 1]],
    [[1, 0], [2, 0], [3, 0], [4, 0]],
]

escape_combinations = [
    [[1, 0], [1, 1]],
    [[1, 0], [2, 0], [2, 1]],
    [[1, 0], [2, 0], [3, 0], [3, 1]],
    [[1, 0], [2, 0], [3, 0], [4, 0]],
]

ADDITIONAL_FEATURES = 17
FEATURE_SIZE = 7 * 7 * 7 + ADDITIONAL_FEATURES

def state_to_features(self, game_state: dict) -> np.array:
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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    """
    #1 1 if way up is obstructed, 0 else
    #2 1 if way right is obstructed, 0 else
    #3 1 if way left is obstructed, 0 else
    #4 1 if way down is obstructed, 0 else
    """
    feature_shape: tuple = (ADDITIONAL_FEATURES,)
    features = np.full(feature_shape, np.nan)

    name, score, is_bomb_possible, (player_x, player_y) = game_state["self"]

    field = game_state["field"]
    coins = np.array(game_state["coins"])

    escape_path = [[0, 0]]

    blast_coords = []

    bomb_pos = [[bomb_x, bomb_y] for (bomb_x, bomb_y), bomb_int in game_state["bombs"]]

    for bomb in game_state["bombs"]:
        blast_coord = get_blast_coords(bomb, field)
        blast_coords += blast_coord

        if (player_x, player_y) not in blast_coord:
            continue

        for route in explosion_escape_combinations:
            escape_array = check_all_paths_for_route(
                np.array(route),
                field,
                player_x,
                player_y,
                bomb_pos,
                check_blast=True,
                blast_coord=blast_coord,
            )

            if np.any(escape_array):
                (escape_int,) = np.where(escape_array)
                escape_path = get_escape_from_path_array(escape_int[0], route)
                break

    can_escape = False

    for route in escape_combinations:
        if check_all_paths_for_route(
            np.array(route), field, player_x, player_y, bomb_pos
        ):
            can_escape = True
            break

    features[0] = (
        np.abs(field[player_x + 1, player_y])
        - 0.5
        + int((player_x + 1, player_y) in bomb_pos)
    )
    features[1] = (
        np.abs(field[player_x - 1, player_y])
        - 0.5
        + int((player_x - 1, player_y) in bomb_pos)
    )
    features[2] = (
        np.abs(field[player_x, player_y + 1])
        - 0.5
        + int((player_x, player_y + 1) in bomb_pos)
    )
    features[3] = (
        np.abs(field[player_x, player_y - 1])
        - 0.5
        + int((player_x, player_y - 1) in bomb_pos)
    )

    features[4] = int(can_escape) - 0.5
    features[5] = int(escape_path[0][0] == 1) - 0.5
    features[6] = int(escape_path[0][1] == 1) - 0.5
    features[7] = int(escape_path[0][0] == -1) - 0.5
    features[8] = int(escape_path[0][1] == -1) - 0.5
    features[9] = int(is_bomb_possible) - 0.5
    features[10] = int((player_x, player_y) in blast_coords) - 0.5

    features[11] = int((player_x + 1, player_y) in blast_coords) - 0.5
    features[12] = int((player_x - 1, player_y) in blast_coords) - 0.5
    features[13] = int((player_x, player_y + 1) in blast_coords) - 0.5
    features[14] = int((player_x, player_y - 1) in blast_coords) - 0.5
    features[15] = coin_degree(player_x, player_y, coins) - 0.5
    features[16] = int(bomb_destroys_crate(player_x, player_y, field)) - 0.5

    view_port = view_port_state(game_state)
    # reduced_map = self.pca.transform(view_port.reshape(1, -1)).reshape(-1)
    reduced_map = view_port
    
    conc= np.concatenate([features, reduced_map]).reshape((-1))
    # make this a torch tensor
    return conc

def is_path_free(
    path, fields, position_x, position_y, bomb_pos, check_blast=False, blast_coord=None
):
    if check_blast:
        if (position_x + path[-1][0], position_y + path[-1][1]) in blast_coord:
            return False
    for x, y in path:
        if (
            fields[position_x + x, position_y + y] == 0
            and [position_x + x, position_y + y] not in bomb_pos
        ):
            continue
        else:
            return False
    return True


def check_all_paths_for_route(
    route, fields, position_x, position_y, bomb_pos, check_blast=False, blast_coord=None
):
    first_neg_route = np.copy(route)
    first_neg_route[:, 0] = -first_neg_route[:, 0]
    args = (fields, position_x, position_y, bomb_pos, check_blast, blast_coord)
    logic_array = np.array(
        [
            # Equal sign routes
            is_path_free(route, *args),
            is_path_free(-route, *args),
            is_path_free(route[:, ::-1], *args),
            is_path_free(-route[:, ::-1], *args),
            # Partial Negative Routes:
            is_path_free(first_neg_route, *args),
            is_path_free(-first_neg_route, *args),
            is_path_free(first_neg_route[:, ::-1], *args),
            is_path_free(-first_neg_route[:, ::-1], *args),
        ]
    )
    if check_blast:
        return logic_array

    return np.any(logic_array)


def bomb_destroys_crate(position_x, position_y, field):
    blast_coord = get_blast_coords(((position_x, position_y), 0), field)
    for coord in blast_coord:
        if field[coord] == 1:
            return True
    return False


def get_escape_from_path_array(route_int, route):
    route = np.array(route)
    first_neg_route = np.copy(route)
    first_neg_route[:, 0] = -first_neg_route[:, 0]
    if route_int == 0:
        return route
    elif route_int == 1:
        return -route
    elif route_int == 2:
        return route[:, ::-1]
    elif route_int == 3:
        return -route[:, ::-1]
    elif route_int == 4:
        return first_neg_route
    elif route_int == 5:
        return -first_neg_route
    elif route_int == 6:
        return first_neg_route[:, ::-1]
    elif route_int == 7:
        return -first_neg_route[:, ::-1]


def get_blast_coords(bomb, fields):
    """Get Blast Coordinates

    :param bomb:
    :param fields:

    :returns: numpy array with coordinate tuples
    """
    coordinates, timer = bomb
    bomb_obj = Bomb(coordinates, "agent", timer, settings.BOMB_POWER, "DUNKELROT!")
    return bomb_obj.get_blast_coords(fields)


def coin_degree(position_x, position_y, coins):
    if coins.size == 0:
        return 1
    my_position = np.array([position_x, position_y])
    # Calculate closest coins angle
    closest_coin_ind = np.argmin(np.linalg.norm(coins - my_position, axis=1))
    closest_coin = coins[closest_coin_ind]

    # angle in radians
    radians = (
        math.atan2(my_position[1] - closest_coin[1], my_position[0] - closest_coin[0])
        + np.pi
    )

    # Only consider 8 directions in which coin lies
    coin_degree = np.degrees(radians) // 45
    if coin_degree == 8:
        coin_degree = 0

    return coin_degree / 8