
import numpy as np
def create_features(game_state):
    # Extracting basic information
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']

    # 1. Distance to Nearest Coin
    distances_to_coins = [np.abs(x-cx) + np.abs(y-cy) for (cx, cy) in coins]
    nearest_coin_distance = min(distances_to_coins) if coins else -1

    # 2. Is Dead End
    directions = [(x+dx, y+dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    free_spaces = sum(1 for d in directions if arena[d] == 0)
    is_dead_end = 1 if free_spaces == 1 else 0

    # 3. Nearby Bomb Threat and 4. Bomb's Time to Explosion
    bomb_threat = 0
    time_to_explode = 5  # Assuming max time is 4 for a bomb to explode
    for (bx, by), t in bombs:
        if abs(bx - x) < 4 or abs(by - y) < 4:
            bomb_threat = 1
            time_to_explode = min(time_to_explode, t)

    # 5. Can Drop Bomb
    can_drop_bomb = 1 if bombs_left > 0 and (x, y) not in bomb_xys else 0

    # 6. Is Next to Opponent
    is_next_to_opponent = 1 if any(abs(ox-x) + abs(oy-y) == 1 for (ox, oy) in others) else 0

    # 7. Is on Bomb
    is_on_bomb = 1 if (x, y) in bomb_xys else 0

    # 8. Number of Crates Nearby
    crates_nearby = sum(1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)] if arena[x+dx, y+dy] == 1)

    # 9. Escape Route Available (simplified for brevity)
    escape_route_available = 1 if free_spaces > 1 else 0

    # 10. Direction to Nearest Target (simplified for brevity)
    # Assuming the function look_for_targets returns a direction as (dx, dy)
    target_direction = look_for_targets(arena == 0, (x, y), coins + others)
    direction_to_target = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
    if target_direction:
        if target_direction == (x, y-1): direction_to_target[0] = 1
        elif target_direction == (x, y+1): direction_to_target[1] = 1
        elif target_direction == (x-1, y): direction_to_target[2] = 1
        elif target_direction == (x+1, y): direction_to_target[3] = 1

    # 11. Is in a Loop (simplified for brevity)
    is_in_loop = 1 if self.coordinate_history.count((x, y)) > 2 else 0

    # 12. Ignore Others Timer (normalized)
    ignore_others_timer_normalized = self.ignore_others_timer / 5  # Assuming max timer is 5

    # Combining all features into a single list
    features = [
        nearest_coin_distance, is_dead_end, bomb_threat, time_to_explode,
        can_drop_bomb, is_next_to_opponent, is_on_bomb, crates_nearby,
        escape_route_available
    ] + direction_to_target + [is_in_loop, ignore_others_timer_normalized]

    return np.array(features)
