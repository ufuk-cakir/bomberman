

from .feature_selection import * # includes state_to_features
import os
import pickle
import random

from .ppo import PPO,HYPER
import torch
import math

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


SIZE_OF_STATE_VECTOR = 360

CONTINUE_TRAINING = 0


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
            self.model = PPO(NUMBER_OF_POSSIBLE_ACTIONS=len(ACTIONS), SIZE_OF_STATE_VECTOR=SIZE_OF_STATE_VECTOR)
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
        self.logger.debug("Choosing action purely at random.")
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


# is imported from feature_selection.py
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
    """
    # This is the dict before the game begins and after it end
   