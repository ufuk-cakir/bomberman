import logging
from pathlib import Path

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 17
ROWS = 17
SCENARIOS = {
    # modes useful for agent development
	"empty": {
        "CRATE_DENSITY": 0, 
        "COIN_COUNT": 0 
    },
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 50
    },
    "loot-crate": { 
        "CRATE_DENSITY": 0.75, 
        "COIN_COUNT": 50 
    }, 
    # this is the tournament game mode
    "classic": {
        "CRATE_DENSITY": 0.75,
        "COIN_COUNT": 9
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}

MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2



# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1


