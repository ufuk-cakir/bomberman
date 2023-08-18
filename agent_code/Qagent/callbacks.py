import os
import pickle
import random

from .Qmodel import QNet, Memory, Transition, HYPER, ACTIONS
from collections import namedtuple
import torch

import numpy as np
import math








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
    
    
    
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
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
    s = state_to_features(self,game_state)
    s = torch.tensor(s, device=self.device, dtype=torch.float).unsqueeze(0)
    
     
    # Epsilon greedy policy
    sample = random.random() # sample from uniform distribution
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


import torch

def state_to_features(self, game_state: dict) -> torch.Tensor:
    """
    Converts the game state to the input of your model, i.e. a feature tensor.

    :param game_state: A dictionary describing the current game board.
    :return: torch.Tensor
    """
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
