import numpy as np
from random import shuffle
from .Settings import *
from collections import deque
from .Events import *
import neat


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"]


class Peacful_Agent:
    def __init__(self):
        np.random.seed()
        self.x : int
        self.y : int
        self.events = []
        self.bombs_left = True
        self.score: int = 0
        self.dead: bool = False
        self.train = False
        self.round = 0

    def get_state(self):
        return Peacful_Agent, self.round, self.score, self.bombs_left, (self.x, self.y), self.dead
    
    def update_score(self, item):
        if item == REWARD_COIN:
            self.score += 1
        elif item == REWARD_KILL: 
            self.score += 5

    def act(self, game_state):
        return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
    
    def add_event(self, event):
        self.events.append(event)



class Coin_Collector_Agent:
    def __init__(self):
        """Called once before a set of games to initialize data structures etc.

        The 'self' object passed to this method will be the same in all other
        callback methods. You can assign new properties (like bomb_history below)
        here or later on and they will be persistent even across multiple games.
        You can also use the self.logger object at any time to write to the log
        file for debugging (see https://docs.python.org/3.7/library/logging.html).
        """
        np.random.seed()
        self.x : int
        self.y : int
        self.events = []
        self.bombs_left = True
        self.score: int = 0
        self.dead: bool = False
        self.train = False
        self.round = 0

    def get_state(self):
        return Coin_Collector_Agent, self.round, self.score, self.bombs_left, (self.x, self.y), self.dead


    def add_event(self, event):
        self.events.append(event)

    
    def update_score(self, item):
        if item == REWARD_COIN:
            self.score += 1
        elif item == REWARD_KILL: 
            self.score += 5


    def look_for_targets(free_space, start, targets):
        """Find direction of the closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards the closest target or towards tile closest to any target.
        """
        if len(targets) == 0:
            return None

        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]


    def act(self, game_state):
        """
        Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        """
        # Gather information about the game state
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if bombs_left > 0:
            valid_actions.append('BOMB')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                    and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates

        # Exclude targets that are currently occupied by a bomb
        targets = [target for target in targets if target not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        for o in others:
            free_space[o] = False
        d = self.look_for_targets(free_space, (x, y), targets)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) <= BOMB_POWER):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) <= BOMB_POWER):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                return a



class Rule_Based_Agent:
    def __init__(self) -> None:
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0
        self.current_round = 0
        self.x : int
        self.y : int
        self.events = []
        self.bombs_left = True
        self.score: int = 0
        self.dead: bool = False
        self.train = False
        self.round = 0

    
    def get_state(self):
        return Rule_Based_Agent, self.round, self.score, self.bombs_left, (self.x, self.y), self.dead

    
    def add_event(self, event):
        self.events.append(event)


    def look_for_targets(self,free_space, start, targets):
        """Find direction of closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards closest target or towards tile closest to any target.
        """
        if len(targets) == 0: return None

        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]


    def reset_self(self):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0

    def update_score(self, item):
        if item == REWARD_COIN:
            self.score += 1
        elif item == REWARD_KILL: 
            self.score += 5


    def act(self, game_state):
        """
        Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        """    
        # Gather information about the game state
        arena = game_state['field']
        _, round, score, bombs_left, (x, y), dead = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, round, s, b, xy, dead) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                    and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = self.look_for_targets(free_space, (x, y), targets)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a == 'BOMB':
                    self.bomb_history.append((x, y))

                return a



class Neat_Agent:
    def __init__(self, genome, config) -> None:
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome,config)
        self.x : int
        self.y : int
        self.events = []
        self.bombs_left = True
        self.score: int = 0
        self.dead: bool = False
        self.train = True
        self.round = 0
        self.coordinate_history = deque([], 20)


    def add_event(self, event):
        self.events.append(event)

    def get_state(self):
        return Neat_Agent, self.round, self.score, self.bombs_left, (self.x, self.y), self.dead
    
    '''def state_to_feature(self, game_state:dict):
        """
        Converts game state with varying size to features of fixed size 1180"""

        game_state_map = np.zeros((17,17))

        game_state_map[game_state['field'] == -1] = -1
        game_state_map[game_state['field'] == 1] == 1

        for (x,y) in game_state['coins']:
            game_state_map[x,y] = 4

        for (type, round, score, bombs, (x,y), dead) in game_state['others']:
            if not dead and bombs:
                game_state_map[x,y] = 2
            if not dead and not bombs:
                game_state_map[x,y] = 3
        
        for (x,y),t in game_state['bombs']:
            game_state_map[x,y] = 5 + t
        
        for x in range(game_state['explosion_map'].shape[0]):
            for y in range(game_state['explosion_map'].shape[1]):
                if game_state['explosion_map'][x,y] > 0:
                    game_state_map[x,y] = 10 + game_state['explosion_map'][x,y]

    
        self_state = np.array([int(game_state['self'][3]), game_state['self'][4][0], game_state['self'][4][1]])

        features = np.concatenate((self_state,  game_state_map.flatten()), axis = None)
        
        return features'''



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
        agent_type, round,  score, bombs_left, (x, y), dead = game_state['self']
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
        is_in_loop = 1 if self.coordinate_history.count((x, y)) > 3 else 0
        self.coordinate_history.append((x, y))

        
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
        
    def update_score(self, item):
        if item == REWARD_COIN:
            self.score = self.score + 1
        elif item == REWARD_KILL: 
            self.score = self.score + 5
    
    def act(self, game_state):
        features = self.state_to_features(game_state)
        output = self.net.activate(features)
        return ACTIONS[output.index(max(output))]
        


class NN_Agent:
    # give path to saved network, load and act upon its decisions
    def __init__(self) -> None:
        self.x : int
        self.y : int
        self.events = []
        self.bombs_left = True
        self.score: int = 0
        self.dead: bool = False
        self.train = False
        self.round = 0

    def get_state(self):
        return NN_Agent, self.round, self.score, self.bombs_left, (self.x, self.y), self.dead


    def update_score(self, item):
        if item == REWARD_COIN:
            self.score += 1
        elif item == REWARD_KILL: 
            self.score += 5


    def add_event(self, event):
        self.events.append(event)

    
    def act(self, game_state):
        pass