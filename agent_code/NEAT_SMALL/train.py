import numpy as np
import pickle
import neat
import os
import copy
import sys
import visualize
from Training_Game import Game, Agents, Events as e



GAME_REWARDS = {'empty' : {e.COIN_COLLECTED: 0.0, e.KILLED_OPPONENT: 0.0, e.KILLED_SELF: 0.0, e.INVALID_ACTION: -1.0, e.WAITED: 0.0,
                                   e.MOVED_LEFT: 20.0, e.MOVED_RIGHT: 20.0, e.MOVED_UP: 20.0, e.MOVED_DOWN: 20.0, e.BOMB_DROPPED: 0.0, 
                                   e.BOMB_EXPLODED: 0.0, e.CRATE_DESTROYED: 0.0, e.COIN_FOUND: 0.0, e.SURVIVED_ROUND: 0.0, e.GOT_KILLED: 0.0},
                'coin-heaven' : {e.COIN_COLLECTED: 120.0, e.KILLED_OPPONENT: 0.0, e.KILLED_SELF: 0.0, e.INVALID_ACTION: -1.0, e.WAITED: 0.0,
                                   e.MOVED_LEFT: 20.0, e.MOVED_RIGHT: 20.0, e.MOVED_UP: 20.0, e.MOVED_DOWN: 20.0, e.BOMB_DROPPED: 0.0, 
                                   e.BOMB_EXPLODED: 0.0, e.CRATE_DESTROYED: 0.0, e.COIN_FOUND: 0.0, e.SURVIVED_ROUND: 0.0, e.GOT_KILLED: 0.0},
                'loot-crate' : {e.COIN_COLLECTED: 2.0, e.KILLED_OPPONENT: 6.0, e.KILLED_SELF: -2.0, e.INVALID_ACTION: -0.5, e.WAITED: -0.1,
                                   e.MOVED_LEFT: 0.3, e.MOVED_RIGHT: 0.3, e.MOVED_UP: 0.3, e.MOVED_DOWN: 0.3, e.BOMB_DROPPED: 0.5, 
                                   e.BOMB_EXPLODED: 0.15, e.CRATE_DESTROYED: 0.5, e.COIN_FOUND: 1.0, e.SURVIVED_ROUND: 5.0, e.GOT_KILLED: -2.0},
                'classic' : {e.COIN_COLLECTED: 2.0, e.KILLED_OPPONENT: 6.0, e.KILLED_SELF: -2.0, e.INVALID_ACTION: -0.5, e.WAITED: -0.1,
                                   e.MOVED_LEFT: 0.3, e.MOVED_RIGHT: 0.3, e.MOVED_UP: 0.3, e.MOVED_DOWN: 0.3, e.BOMB_DROPPED: 0.5, 
                                   e.BOMB_EXPLODED: 0.15, e.CRATE_DESTROYED: 0.5, e.COIN_FOUND: 1.0, e.SURVIVED_ROUND: 5.0, e.GOT_KILLED: -2.0}}


def calc_genome_fitness(events, scenario):
    fitness = 0
    for event in events:
        fitness += GAME_REWARDS[scenario][event]
    return fitness


def eval_genomes(genomes, config):
    ## for each training run, create one game with agents
    agents = []
    scenario = 'coin-heaven'
    rounds = 10
    game = Game.Game(agents, scenario, rounds)

    for i, (genome_id1, genome1) in enumerate(genomes):
        # train with created world and agents
        g = copy.deepcopy(game)
        agent = Agents.Neat_Agent(genome1, config)
        g.play(agent)
        genome1.fitness = agent.score #calc_genome_fitness(agent_events, scenario) #train_genome(genome1, config, game, arena) 



def train_neat(config):
    ## load poplulation from config file or from checkpoint
    #files = os.listdir()
    #if 'neat-checkpoint-xx'
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-26')
    p = neat.Population(config)

    ## print generation information to screen
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    

    winner = p.run(eval_genomes, 100)
    
    with open("winner.nt", "wb") as f:
        pickle.dump(winner,f)

    #visualize.draw_net(config, winner, True)
    #visualize.draw_net(config, winner, True, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)



def setup_training():
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # load network if available
    # train agents and get best save it and exit code 
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    train_neat(config)
    return 


if __name__ == '__main__':
    setup_training()