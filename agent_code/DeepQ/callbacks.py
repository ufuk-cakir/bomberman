

import os
import pickle
import random

from .model import deepQ,HYPER
import torch
import math

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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

    if self.train:
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
    self.steps_done = 0
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.logger.info("Using device: {}".format(self.device))
    if self.train or not os.path.isfile(HYPER.model_name):
        if CONTINUE_TRAINING:
            self.logger.info("Loading model from saved state.")
            with open(HYPER.model_name, "rb") as file:
                self.model = pickle.load(file)

        else:
            self.logger.info("Setting up model from scratch.")
            self.model = deepQ(input_channels=HYPER.input_channels, num_actions = len(ACTIONS), device=self.device)
            
    else:
        with open(HYPER.model_name, "rb") as file:
            self.model = pickle.load(file)
        self.logger.info("Loading model from saved state.")
       


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
    s = torch.tensor(s, device=self.device, dtype=torch.float)
    s = s.unsqueeze(0)
    # Epsilon greedy
    sample = random.random()
    eps_threshold = HYPER.epsilon_end + (HYPER.epsilon_start - HYPER.epsilon_end) * \
        math.exp(-1. * self.steps_done / HYPER.epsilon_decay)
        
    self.steps_done += 1
   
    if sample > eps_threshold or not self.train:
        with torch.no_grad():
            model_querry = self.model.policy_net(s).max(1)[1].view(1, 1)
            return ACTIONS[model_querry]
    else:
        return np.random.choice(ACTIONS)

    

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
    
    height, width = arena.shape
    
    terrain_channel = np.zeros((width, height))
    bomb_channel = np.zeros((width, height))
    explosion_channel = np.zeros((width, height))
    coin_channel = np.zeros((width, height))
    self_channel = np.zeros((width, height))
    enemy_channel = np.zeros((width, height))
    
    # Create channels for each type of object
    terrain_channel = arena.T # Transpose to get correct orientation
    for (x, y), countdown in game_state['bombs']:
        bomb_channel[x, y] = countdown

    # Fill in explosion data
    explosion_channel = game_state['explosion_map'].T

    # Fill in coin data
    for (x, y) in game_state['coins']:
        coin_channel[x, y] = 1

    # Fill in self data
    self_x, self_y = game_state['self'][-1]
    self_channel[self_x, self_y] = 1

    # Fill in enemy data
    for _, _, _, (enemy_x, enemy_y) in game_state['others']:
        enemy_channel[enemy_x, enemy_y] = 1

    # Stack all channels together
    stacked_channels = np.stack([terrain_channel, bomb_channel, explosion_channel, coin_channel, self_channel, enemy_channel])
    print(coin_channel)
    return stacked_channels
        
    
    