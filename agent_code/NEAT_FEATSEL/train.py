import numpy as np
import pickle
import neat
import os
import copy
import sys
import visualize
from collections import Counter
import json
from Training_Game import Game, Agents, Events as e

NUM_GENERATIONS = 5000

GAME_REWARDS = {'coin-heaven' : {e.COIN_COLLECTED: 6.0, e.KILLED_OPPONENT: 0.0, e.KILLED_SELF: -2.0, e.INVALID_ACTION: -5.0, e.WAITED: -0.1,
                                   e.MOVED_LEFT: -0.1, e.MOVED_RIGHT: -0.1, e.MOVED_UP: -0.1, e.MOVED_DOWN: -0.1, e.BOMB_DROPPED: -0.1, 
                                   e.BOMB_EXPLODED: 0.0, e.CRATE_DESTROYED: 0.0, e.COIN_FOUND: 0.0, e.SURVIVED_ROUND: 0.0, e.GOT_KILLED: 0.0,
                                   e.CLOSER_TO_COIN: 3.0, e.FURTHER_FROM_COIN: -1.0, e.IS_IN_LOOP: -1.0, e.IN_BLAST_RADIUS: -0.5, e.GOING_AWAY_FROM_BOMB: 0.5,
                                   e.GOING_TOWARDS_BOMB: -0.5, e.ESCAPED_BOMB: 2.0},
                'loot-crate' : {e.COIN_COLLECTED: 2.0, e.KILLED_OPPONENT: 6.0, e.KILLED_SELF: -2.0, e.INVALID_ACTION: -0.5, e.WAITED: -0.1,
                                   e.MOVED_LEFT: 0.3, e.MOVED_RIGHT: 0.3, e.MOVED_UP: 0.3, e.MOVED_DOWN: 0.3, e.BOMB_DROPPED: 0.5, 
                                   e.BOMB_EXPLODED: 0.15, e.CRATE_DESTROYED: 0.5, e.COIN_FOUND: 1.0, e.SURVIVED_ROUND: 5.0, e.GOT_KILLED: -2.0},
                'classic' : {e.COIN_COLLECTED: 2.0, e.KILLED_OPPONENT: 6.0, e.KILLED_SELF: -2.0, e.INVALID_ACTION: -0.5, e.WAITED: -0.1,
                                   e.MOVED_LEFT: 0.3, e.MOVED_RIGHT: 0.3, e.MOVED_UP: 0.3, e.MOVED_DOWN: 0.3, e.BOMB_DROPPED: 0.5, 
                                   e.BOMB_EXPLODED: 0.15, e.CRATE_DESTROYED: 0.5, e.COIN_FOUND: 1.0, e.SURVIVED_ROUND: 5.0, e.GOT_KILLED: -2.0}}

generation_events = []


def calc_genome_fitness(agent, scenario):
    ## function to calculate a genomes fitness based on the scenario and the agent
    fitness = 0
    for event in agent.events:
        fitness += GAME_REWARDS[scenario][event]
    return fitness


def eval_genomes(genomes, config):
    ## for each training run, create one game with agents, perform fitness evaluation, record events for generation
    agents = []
    scenario = 'coin-heaven'
    rounds = 5
    game = Game.Game(agents, scenario, rounds)
    population_events = []


    for i, (genome_id1, genome1) in enumerate(genomes):
        # train with created world and agents
        g = copy.deepcopy(game)
        agent = Agents.Neat_Agent(genome1, config)
        g.play(agent)
        genome1.fitness = calc_genome_fitness(agent, scenario) 
        population_events = population_events + agent.events
    print({k: v for k, v in sorted(dict(Counter(population_events)).items(), key=lambda item: item[1], reverse=True)})
    generation_events.append(population_events)


def train_neat(config):
    ## initialize training with population and set reporter for command line output, save winner when training over
    p = neat.Population(config)

    ## print generation information to screen
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))

    

    winner = p.run(eval_genomes, NUM_GENERATIONS)
    with open("winner.nt", "wb") as f:
        pickle.dump(winner,f)

    stats.save_genome_fitness()
    stats.save_species_count()
    stats.save_species_fitness()
    generation_event_dict = {}
    for i in range(len(generation_events)):
        generation_event_dict[i] = dict(Counter(generation_events[i]))

    with open("generation_events.json", "w") as outfile:
        json.dump(generation_event_dict, outfile)
           



def setup_training():
    ## once called for setting up training, return local config file 
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    return config


if __name__ == '__main__':
    config = setup_training()
    train_neat(config)