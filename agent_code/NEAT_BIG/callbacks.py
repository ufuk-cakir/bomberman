import os
import pickle
import random
import numpy as np
import neat



ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"]


SIZE_OF_STATE_VECTOR = 316

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
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = PPO2(HYPER)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    """
    with open("winner.nt", "rb") as f:
        winner = pickle.load(f)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    winner_network = neat.nn.FeedForwardNetwork.create(winner,config)


    self.model = winner_network
    return 

def state_to_feature(game_state:dict):
    """
    Converts game state with varying size to features of fixed size 1180"""
    field_map = game_state['field']
    bomb_map = np.zeros((17,17))
    coin_map = np.zeros((17,17))
    explosion_map = np.zeros((17,17))
    others_map = np.zeros((17,17))

    for (xpos,ypos),t in game_state['bombs']:
        bomb_map[xpos, ypos] = t
    
    for (xpos, ypos) in game_state['coins']:
        coin_map[xpos,ypos] = 1
        
    for agent_state in game_state['others']:
        if agent_state[2] == 1:
            others_map[agent_state[3][0], agent_state[3][1]] = 1
        else:
            others_map[agent_state[3][0], agent_state[3][1]] = -1

    self_state = game_state['self']
    player_states = np.array([int(self_state[2]), self_state[3][0], self_state[3][1]])

    features = np.concatenate((field_map.flatten(), bomb_map.flatten(), explosion_map.flatten(), coin_map.flatten(), others_map.flatten(), player_states), axis = None)
    
    return features


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_feature(game_state)
    output = self.model.activate(features)
    return ACTIONS[int(max(output))]

    # return action
    
