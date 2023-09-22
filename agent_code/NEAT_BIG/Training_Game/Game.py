import numpy as np
import copy
from typing import List, Tuple, Dict
from .Items import Coin, Explosion, Bomb
from .Settings import *
from .Events import *
from .Agents import *

class Game:
    running: bool = True
    step :int = 0
    agents: List = []
    active_agents: List = []
    arena: np.ndarray
    coins: List[Coin] = []
    bombs: List[Bomb] = []
    explosions: List[Explosion] = []
    rounds: int
    round_id: int
   
    def __init__(self, agents, rounds = 1):
        self.agents = agents
        self.rng = np.random.default_rng()
        self.initialize_new_game()
        self.rounds = rounds

    def initialize_new_game(self):
        WALL = -1
        FREE = 0
        CRATE = 1
        arena = np.zeros((COLS, ROWS), int)

        scenario_info = SCENARIOS['classic']

        # Crates in random locations
        arena[self.rng.random((COLS, ROWS)) < scenario_info["CRATE_DENSITY"]] = CRATE

        # Walls
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL
        for x in range(COLS):
            for y in range(ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = WALL

        # Clean the start positions
        start_positions = [(1, 1), (1, ROWS - 2), (COLS - 2, 1), (COLS - 2, ROWS - 2)]
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if arena[xx, yy] == 1:
                    arena[xx, yy] = FREE

        # Place coins at random, at preference under crates
        coins = []
        all_positions = np.stack(np.meshgrid(np.arange(COLS), np.arange(ROWS), indexing="ij"), -1)
        crate_positions = self.rng.permutation(all_positions[arena == CRATE])
        free_positions = self.rng.permutation(all_positions[arena == FREE])
        coin_positions = np.concatenate([
            crate_positions,
            free_positions
        ], 0)[:scenario_info["COIN_COUNT"]]
        for x, y in coin_positions:
            coins.append(Coin((x, y), collectable=arena[x, y] == FREE))

        # Reset agents and distribute starting positions
        active_agents = []
        for agent, start_position in zip(self.agents, self.rng.permutation(start_positions)):
            active_agents.append(agent)
            agent.x, agent.y = start_position

        self.arena = arena
        self.coins = coins
        self.active_agents = active_agents
        self.bombs = []
        self.explosions = []
        return 
    
    def get_state_for_agent(self, agent):
        if agent.dead:
            return None

        state = {
            'step': self.step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            if exp.is_dangerous():
                for (x, y) in exp.blast_coords:
                    explosion_map[x, y] = max(explosion_map[x, y], exp.timer - 1)
        state['explosion_map'] = explosion_map

        return state
    
    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free
    
    def perform_agent_action(self, agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left:
            self.bombs.append(Bomb((agent.x, agent.y), agent, BOMB_TIMER, BOMB_POWER))
            agent.bombs_left = False
            agent.add_event(BOMB_DROPPED)
        elif action == 'WAIT':
            agent.add_event(WAITED)
        else:
            agent.add_event(INVALID_ACTION)


    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        a.update_score(REWARD_COIN)
                        a.add_event(COIN_COLLECTED)

    def update_explosions(self):
        # Progress explosions
        remaining_explosions = []
        for explosion in self.explosions:
            explosion.timer -= 1
            if explosion.timer <= 0:
                explosion.next_stage()
                if explosion.stage == 1:
                    explosion.owner.bombs_left = True
            if explosion.stage is not None:
                remaining_explosions.append(explosion)
        self.explosions = remaining_explosions

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer <= 0:
                # Explode when timer is finished
                bomb.owner.add_event(BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.add_event(CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                bomb.owner.add_event(COIN_FOUND)

                # Create explosion
                screen_coords = [(GRID_OFFSET[0] + GRID_SIZE * x, GRID_OFFSET[1] + GRID_SIZE * y) for (x, y) in
                                 blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner, EXPLOSION_TIMER))
                bomb.active = False
            else:
                # Progress countdown
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.is_dangerous():
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            a.add_event(KILLED_SELF)
                        else:
                            explosion.owner.update_score(REWARD_KILL)
                            explosion.owner.add_event(KILLED_OPPONENT)

        # Remove hit agents
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.add_event(GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.add_event(OPPONENT_ELIMINATED)


    def time_to_stop(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            self.running = False
            return True

        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            self.running = False
            return True

        if any(a.train for a in self.agents):
            if not any([a.train for a in self.active_agents]):
                self.running = False
                return True

        if self.step >= MAX_STEPS:
            self.running = False
            return True

        return False
    
    def poll_and_run_agents(self):
        # Tell agents to act
        np.random.shuffle(self.active_agents)
        for a in self.active_agents:
            state = self.get_state_for_agent(a)
            action = a.act(state)
            self.perform_agent_action(a, action)

    def get_start_position(self):
        start_positions = [(1, 1), (1, ROWS - 2), (COLS - 2, 1), (COLS - 2, ROWS - 2)]
        taken_start_positions = [(agent.x, agent.y) for agent in self.agents]
        remaining_start_position = list(set(start_positions) - set(taken_start_positions))
        return remaining_start_position[0][0], remaining_start_position[0][1]
    
    def reset_round(self):
        for a in self.agents:
            a.round += 1
            a.score = 0
            a.dead = False
            a.bombs_left = True
        self.initialize_new_game()
        

    def play(self, neat_agent):
        neat_agent.x, neat_agent.y = self.get_start_position()
        self.agents.append(neat_agent)
        self.active_agents.append(neat_agent)

        for i in range(self.rounds):
            while self.running:
                self.poll_and_run_agents()
                self.collect_coins()
                self.update_explosions()
                self.evaluate_explosions()
                self.update_bombs()
                self.time_to_stop()
                self.step += 1
            
            # Clean up survivors and in
            for a in self.active_agents:
                a.add_event(SURVIVED_ROUND)
                

       

  