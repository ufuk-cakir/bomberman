import numpy as np
import pickle
import neat
import os
import copy
import sys
from Training_Game import Game, Agents, Events as e



GAME_REWARDS = {
    e.COIN_COLLECTED: 5.0,
    e.KILLED_OPPONENT: 8.0,
    e.KILLED_SELF: -2.0,
    e.INVALID_ACTION: -3.0,
    e.WAITED: 0.0,
    e.MOVED_LEFT: 1.0,
    e.MOVED_RIGHT: 1.0,
    e.MOVED_UP: 1.0,
    e.MOVED_DOWN: 1.0,
    e.BOMB_DROPPED: 2.0,
    e.BOMB_EXPLODED: 0.5,
    e.CRATE_DESTROYED: 2.0,
    e.COIN_FOUND: 2.0,
    e.SURVIVED_ROUND: 6.0,
    e.GOT_KILLED: -5.0,
    }


def calc_genome_fitness(events):
    fitness = 0
    for event in events:
        fitness += GAME_REWARDS[event]
    return fitness


def eval_genomes(genomes, config):
    ## for each training run, create one game with agents
    agents = [Agents.Peacful_Agent(), Agents.Peacful_Agent(), Agents.Peacful_Agent()]
    game = Game.Game(agents)
    #Neat_Agent = genome_to_agent(5,config)
    #game.play(Neat_Agent)
    #calc_genome_fitness(Neat_Agent.events)
    #quit()

    for i, (genome_id1, genome1) in enumerate(genomes):
        # train with created world and agents
        g = copy.deepcopy(game)

        agent = Agents.Neat_Agent(genome1, config)

        g.play(agent)

        agent_events = agent.events

        genome1.fitness = calc_genome_fitness(agent_events) #train_genome(genome1, config, game, arena)



def train_neat(config):
    ## load poplulation from config file or from checkpoint
    
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-26')
    p = neat.Population(config)

    ## print generation information to screen
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    

    winner = p.run(eval_genomes, 50)
    
    with open("winner.nt", "wb") as f:
        pickle.dump(winner,f)



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