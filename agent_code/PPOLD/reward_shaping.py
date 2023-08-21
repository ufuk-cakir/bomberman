
import numpy as np
from typing import List
import events as e
from .feature_selection import *
import settings


def custom_rewards(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
   

    """_summary_

   
    game_rewards = {
    e.COIN_COLLECTED: 2.0,
    e.KILLED_OPPONENT: 6.0,
    e.KILLED_SELF: -2.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.1,
    e.MOVED_LEFT: 0.1,
    e.MOVED_RIGHT: 0.1,
    e.MOVED_UP: 0.1,
    e.MOVED_DOWN: 0.1,
    e.BOMB_DROPPED: 0.5,
    e.BOMB_EXPLODED: 0.15,
    e.CRATE_DESTROYED: 0.5,
    e.COIN_FOUND: 1.0,
    e.SURVIVED_ROUND: 5.0,
    e.GOT_KILLED: -2.0,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    
    # Give reward depending on how long the agent survived
    
    
    self.reward_history.append(reward_sum)
    return reward_sum*0.1
   
    """
    game_rewards = {
        e.COIN_COLLECTED: 300,
        e.KILLED_OPPONENT: 700,
        #e.OPPONENT_ELIMINATED: 700,
        e.KILLED_SELF: -400,
        # e.BOMB_DROPPED: 10,
        e.COIN_FOUND: 10,
        e.CRATE_DESTROYED: 10,
        e.INVALID_ACTION: -50,
        # e.SURVIVED_ROUND: 100,
        BOMB_DESTROYS_CRATE: 40,
        ESCAPABLE_BOMB: 75,
        e.MOVED_LEFT: -10,
        e.MOVED_DOWN: -10,
        e.MOVED_RIGHT: -10,
        e.MOVED_UP: -10,
        e.WAITED: -50,
        WAITED_TOO_LONG: -350,
        CLOSER_TO_COIN: 40,
        IN_BLAST_RADIUS: -7,
        # e.BOMB_DROPPED: -1,
        FURTHER_FROM_COIN: -60,
        CLOSER_TO_COIN:70,
    }
    # TODO Reward for going in coin direction
    # TODO add punishment for visiting same fields
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # Useless bomb

    if e.BOMB_DROPPED in events and BOMB_DESTROYS_CRATE not in events:
        reward_sum -= 600
        # print("useless")
    # Inescapable bomb
    if e.BOMB_DROPPED in events and ESCAPABLE_BOMB not in events:
        reward_sum += game_rewards[e.KILLED_SELF]
    # Reward for going out of blast radius
    # TODO: check if this works    
    
    self.reward_history.append(reward_sum)
    return reward_sum*0.1



#-------------------------------- TODO

CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"
WAITED_TOO_LONG = "WAITED_TOO_LONG"
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"

from .callbacks import get_blast_coords

def reward_coin_distance(old_game_state, new_game_state, events):
    if old_game_state is None or new_game_state is None:
        return

    old_pos = np.array(old_game_state["self"][-1])
    new_pos = np.array(new_game_state["self"][-1])
    old_coins = np.array(old_game_state["coins"])
    new_coins = np.array(new_game_state["coins"])

    persistent_coins = old_coins[np.all(np.isin(old_coins, new_coins))]

    if persistent_coins.size == 0:
        return
    else:
        (persistent_coins,) = persistent_coins
    # Calculate old distances to all persistent coins
    old_distances = np.sqrt(np.sum((persistent_coins - old_pos) ** 2, axis=-1))
    # Find the index of the previous closest coin
    closest_coin_index = np.argmin(old_distances)
    # Get the distance to this previously closest coin now
    new_distance = np.sqrt(
        np.sum(((persistent_coins[closest_coin_index] - new_pos) ** 2))
    )
    # Also get the old distance.
    # Should be the minimum of `old_distances`.
    old_distance = old_distances[closest_coin_index]
    if new_distance > old_distance:
        events.append(FURTHER_FROM_COIN)

    elif new_distance < old_distance:
        events.append(CLOSER_TO_COIN)


def punish_long_wait(waited_for, events):
    max_wait = settings.EXPLOSION_TIMER
    if e.WAITED in events:
        self.waited_for += 1
    else:
        self.waited_for = 0
    if self.waited_for > max_wait:
        events.append(WAITED_TOO_LONG)


def check_placed_bomb(old_features, new_game_state, events):
    escapable_bomb = False
    destroy_crate = False
    field = new_game_state["field"]
    name, score, is_bomb_possible, (player_x, player_y) = new_game_state["self"]
    if "BOMB_DROPPED" in events:
        for bomb in new_game_state["bombs"]:
            (bomb_x, bomb_y), timer = bomb
            if [bomb_x, bomb_y] == [player_x, player_y]:
                if old_features is not None:
                    escapable_bomb = bool(old_features[0][4] > 0)
                blast_coord = get_blast_coords(bomb, field)
                for coord in blast_coord:
                    if field[coord] == 1:
                        destroy_crate = True
                        return destroy_crate, escapable_bomb
    return destroy_crate, escapable_bomb


def check_blast_radius(game_state, events):
    fields = game_state["field"]
    _, _, _, (player_x, player_y) = game_state["self"]
    for bomb in game_state["bombs"]:
        blast_coord = get_blast_coords(bomb, fields)
        if (player_x, player_y) in blast_coord:
            events.append(IN_BLAST_RADIUS)
            return