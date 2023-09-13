import os
import pickle
import random
import numpy as np
import neat

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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

    game_state_map = np.zeros((17,17))

    game_state_map[game_state['field'] == -1] = -1
    game_state_map[game_state['field'] == 1] == 1

    for (x,y) in game_state['coins']:
        game_state_map[x,y] = 4

    for (name, score, bomb, (x,y)) in game_state['others']:
        if bomb:
            game_state_map[x,y] = 2
        else:
            game_state_map[x,y] = 3
    
    for (x,t),t in game_state['bombs']:
        game_state_map[x,y] = 5 + t
    
    for x in range(game_state['explosion_map'].shape[0]):
        for y in range(game_state['explosion_map'].shape[1]):
            if game_state['explosion_map'][x,y] > 0:
                game_state_map[x,y] = 10 + game_state['explosion_map'][x,y]


    self_state = np.array([int(game_state['self'][2]), game_state['self'][3][0], game_state['self'][3][1]])

    features = np.concatenate((self_state,  game_state_map.flatten()), axis = None)
    
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
    return ACTIONS[output.index(max(output))]

    # return action
    
