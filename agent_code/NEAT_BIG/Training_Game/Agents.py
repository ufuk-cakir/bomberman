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


    def add_event(self, event):
        self.events.append(event)

    def get_state(self):
        return Neat_Agent, self.round, self.score, self.bombs_left, (self.x, self.y), self.dead
    

    def state_to_feature(self, game_state:dict):
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
            if agent_state[5]:
                if agent_state[3] == 1:
                    others_map[agent_state[4][0], agent_state[4][1]] = 1
                else:
                    others_map[agent_state[4][0], agent_state[4][1]] = -1

        self_state = game_state['self']
        player_states = np.array([int(self_state[3]), self_state[4][0], self_state[4][1]])

        features = np.concatenate((field_map.flatten(), bomb_map.flatten(), explosion_map.flatten(), coin_map.flatten(), others_map.flatten(), player_states), axis = None)
        
        return features
    
    def update_score(self, item):
        if item == REWARD_COIN:
            self.score += 1
        elif item == REWARD_KILL: 
            self.score += 5
    
    def act(self, game_state):
        features = self.state_to_feature(game_state)
        output = self.net.activate(features)
        return ACTIONS[int(max(output))]
        


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