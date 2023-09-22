import os
import pickle
import random
import numpy as np
import neat

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


SIZE_OF_STATE_VECTOR = 48

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
    with open("winner.nt", "rb") as f:
        winner = pickle.load(f)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    winner_network = neat.nn.FeedForwardNetwork.create(winner,config)


    self.model = winner_network
    return 


def state_to_features(game_state: dict) -> np.array:
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
    name, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    explosion_map = game_state['explosion_map'] 
    
    explosion_coords = [(x,y) for x in range(arena.shape[0]) for y in range(arena.shape[1]) if explosion_map[x,y] > 0]

    #Distance to Nearest Coin
    distances_to_coins = [np.abs(x-cx) + np.abs(y-cy) for (cx, cy) in coins]
    nearest_coin_distance = min(distances_to_coins) if coins else -1


    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    
    # Check blast range of each bomb
    blast_coords = []
    blast_coords_timer = []
    
    for (bx, by), t in bombs:
        # Start with the bomb position itself
        blast_coords.append((bx, by))
        blast_coords_timer.append(t)
        # For each direction, check for crates
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: # TODO maybe add diagonal directions
            for i in range(1, 4):
                # Stop when hitting a wall
                if arena[bx + i*dx, by + i*dy] == -1:
                    break
                # Add free spaces
                if arena[bx + i*dx, by + i*dy] == 0:
                    blast_coords.append((bx + i*dx, by + i*dy))
                    blast_coords_timer.append(t)
                # Stop when hitting a crate
                if arena[bx + i*dx, by + i*dy] == 1:
                    blast_coords.append((bx + i*dx, by + i*dy))
                    blast_coords_timer.append(t)
                    break
    

                
    # Check if agent is in bomb blast range
    bomb_threat = 1 if (x, y) in blast_coords else 0
    # additionaly check if agent is in explosion map
    if explosion_map[x, y] > 0:
        bomb_threat = 1


    # Can Drop Bomb
    can_drop_bomb = 1 if bombs_left > 0 and (x, y) not in bomb_xys else 0

    # Is Next to Opponent
    is_next_to_opponent = 1 if any(abs(ox-x) + abs(oy-y) == 1 for (ox, oy) in others) else 0

    # Is on Bomb
    is_on_bomb = 1 if (x, y) in bomb_xys else 0

    # Is in a Loop 
    is_in_loop = 1 #if self.coordinate_history.count((x, y)) > 3 else 0
    #self.coordinate_history.append((x, y))

    
    blast_in_direction = [0, 0, 0, 0] # [UP, DOWN, LEFT, RIGHT]    
    danger_level = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]

    if bomb_threat:
        # Count number of tiles occupied by blast in each direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    
        # Iterate through each direction
        for i, (dx, dy) in enumerate(directions):
            for j in range(1, 4):  # Check up to 3 tiles in each direction
                # Calculate the coordinates of the tile in the current direction
                tile_x, tile_y = x + j*dx, y + j*dy
                # Check if the tile is in the blast range
                if (tile_x, tile_y) in blast_coords:
                    blast_in_direction[i] += 1
                if (tile_x,tile_y) in explosion_coords:
                
                    danger_level[i] += 1
                # If the tile is a wall, stop checking further in this direction
                elif arena[tile_x, tile_y] == -1:
                    break
                
    
    if len(explosion_coords) > 0:
        # Count number of tiles occupied by blast in each direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    
        # Iterate through each direction
        for i, (dx, dy) in enumerate(directions):
            for j in range(1, 4):  # Check up to 3 tiles in each direction
                # Calculate the coordinates of the tile in the current direction
                tile_x, tile_y = x + j*dx, y + j*dy
            
                if (tile_x,tile_y) in explosion_coords:
                    
                    danger_level[i] += 1
                # If the tile is a wall, stop checking further in this direction
                elif arena[tile_x, tile_y] == -1:
                    break
            

        
    

    # Distance to closest bomb
    distance_to_bombs = [np.abs(x-bx) + np.abs(y-by) for (bx, by) in bomb_xys]
    nearest_bomb_distance = min(distance_to_bombs) if bomb_xys else -1
    
    # time to explode
    time_to_explode = 5  # Assuming max time is 4 for a bomb to explode TODO is this correct?
    for (bx, by), t in bombs:
        if abs(bx - x) < 4 or abs(by - y) < 4:
            time_to_explode = min(time_to_explode, t)

    # Get angle to nearest coin
    if len(coins) > 0:
        nearest_coin = coins[np.argmin(distances_to_coins)]
        angle_to_nearest_coin = np.arctan2(nearest_coin[1] - y, nearest_coin[0] - x)
    
    else:
        angle_to_nearest_coin = -1
    
    
    # Add direction to nearest coin: 
    # Encode if the coin is above or below the agent in first bit, and if it is left or right in second bit
    direction_to_coin = [-1, -1]
    if len(coins) > 0:
        nearest_coin = coins[np.argmin(distances_to_coins)]
        if nearest_coin[0] - x > 0:
            direction_to_coin[1] = 1 # Coin is to the right
        else:
            direction_to_coin[1] = 0 # Coin is to the left
            
        if nearest_coin[1] - y > 0:
            direction_to_coin[0] = 1 # Coin is below
        else:
            direction_to_coin[0] = 0 # Coin is above
            
    
    # local map 
    local = np.zeros((5,5)) -2.5
    
    walls = [(xs,ys) for xs in range(arena.shape[0]) for ys in range(arena.shape[1]) if arena[xs,ys] == -1]
    
    # get the coordinates of the local map view
    local_coords = [(xs,ys) for xs in range(x-2,x+3) for ys in range(y-2,y+3)]

    # fill in arena values
    for (xs,ys) in local_coords:
        if (xs,ys) in walls:
            local[xs-x+2,ys-y+2] = -2.5
        elif (xs,ys) in coins:
            local[xs-x+2,ys-y+2] = 1.5
        elif (xs,ys) in crates:
            local[xs-x+2,ys-y+2] = 1
        elif (xs,ys) in others:
            local[xs-x+2,ys-y+2] = -4.5
        elif (xs,ys) in explosion_coords:
            local[xs-x+2,ys-y+2] = -explosion_map[xs,ys]-0.125
        elif (xs,ys) in blast_coords:
            # Fill in with the timer of bomb explosion
            index = blast_coords.index((xs,ys))
            local[xs-x+2,ys-y+2] = -blast_coords_timer[index]-0.125
        elif (xs,ys) == (x,y):
            local[xs-x+2,ys-y+2] = 5
        elif xs < 0 or xs > 16 or ys < 0 or ys > 16:
            local[xs-x+2,ys-y+2] = -2.5
        
        else:
            local[xs-x+2,ys-y+2] = 0
    
    
    # Start with agent position
    local[2,2] = 5 # agent position
    local = local.flatten().tolist()
    

    valid_directions = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [UP, DOWN, LEFT, RIGHT]
    for i, (dx, dy) in enumerate(directions):
        tile_x, tile_y = x + dx, y + dy
        # check if the tile is free
        if arena[tile_x, tile_y] ==0:
            valid_directions[i] = 1
    
    
    features = valid_directions+[
        nearest_coin_distance, angle_to_nearest_coin] + direction_to_coin + [ bomb_threat, time_to_explode,
        can_drop_bomb, is_next_to_opponent, is_on_bomb, 
    ] + [is_in_loop, nearest_bomb_distance] + blast_in_direction + danger_level + local #ignore_others_timer_normalized]
    return(np.array(features))
    



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    output = self.model.activate(features)
    return ACTIONS[output.index(max(output))]

    # return action
    
