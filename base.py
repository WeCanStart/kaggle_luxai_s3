from enum import IntEnum
import numpy as np
import math
import copy
import os
from sys import stderr


class Global:
    # Game related constants:

    STEP = 0
    SPACE_SIZE = 24
    MAX_UNITS = 16
    RELIC_REWARD_RANGE = 2
    MAX_STEPS_IN_MATCH = 100
    MAX_ENERGY_PER_TILE = 20
    MIN_RELIC_NODES = 2
    MAX_RELIC_NODES = 6
    TEAM_ID = None

    # We will find the exact value of these constants during the game
    UNIT_MOVE_COST = 1  # OPTIONS: list(range(1, 6))
    UNIT_SAP_COST = 30  # OPTIONS: list(range(30, 51))
    UNIT_SAP_RANGE = 3  # OPTIONS: list(range(3, 8))
    UNIT_SENSOR_RANGE = 2  # OPTIONS: list(range(2, 5))
    POSSIBLE_OBSTACLE_MOVEMENT_SPEED = np.array([0.025, 0.05, 0.1, 0.15])
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]

    NEBULA_ENERGY_REDUCTION = 0  # OPTIONS: [0, 10, 25]
    NEBULAS_MEET = [0, 0, 0, 0, 0, 0]
    NEBULAS_POSSIBLE = [0, 1, 2, 3, 5, 25]
    LOWER_NEBULA_RANGE_REDUCTION = -1
    UPPER_NEBULA_RANGE_REDUCTION = 8

    DROPOFF = 0.5
    DROPOFFS_MEET = np.array([0, 0, 0])
    DROPOFFS_POSSIBLE = np.array([0.25, 0.5, 1.0])

    RESERCHER_NEED = 1

    POINTS = 0
    POINTS_GAIN = 0
    ENEMY_POINTS = 0
    ENEMY_POINTS_GAIN = 0

    # Exploration flags:

    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = True
    OBSTACLE_MOVEMENT_SPEED_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False

    # Game logs:

    # REWARD_RESULTS: [{"nodes": Set[Node], "points": int}, ...]
    # A history of reward events, where each entry contains:
    # - "nodes": A set of nodes where our ships were located.
    # - "points": The number of points scored at that location.
    # This data will help identify which nodes yield points.
    REWARD_RESULTS = []

    # obstacles_movement_status: list of bool
    # A history log of obstacle (asteroids and nebulae) movement events.
    # - `True`: The ships' sensors detected a change in the obstacles' positions at this step.
    # - `False`: The sensors did not detect any changes.
    # This information will be used to determine the speed and direction of obstacle movement.
    OBSTACLES_MOVEMENT_STATUS = dict([(0, False), (1, True)])

    # Others:

    # The energy on the unknown tiles will be used in the pathfinding
    HIDDEN_NODE_ENERGY = 0


SPACE_SIZE = Global.SPACE_SIZE


class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


_DIRECTIONS = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  #  down
    (-1, 0),  # left
    (0, 0),  # sap
]


class ActionType(IntEnum):
    center = 0
    up = 1
    right = 2
    down = 3
    left = 4
    sap = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def from_coordinates(cls, current_position, next_position):
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]

        if dx < 0:
            return ActionType.left
        elif dx > 0:
            return ActionType.right
        elif dy < 0:
            return ActionType.up
        elif dy > 0:
            return ActionType.down
        else:
            return ActionType.center

    def to_direction(self):
        return _DIRECTIONS[self]


def get_match_step(step: int) -> int:
    return step % (Global.MAX_STEPS_IN_MATCH + 1)


def get_match(step: int) -> int:
    return step // (Global.MAX_STEPS_IN_MATCH + 1)


def warp_int(x):
    if x >= SPACE_SIZE:
        x -= SPACE_SIZE
    elif x < 0:
        x += SPACE_SIZE
    return x


def warp_point(x, y) -> tuple:
    return warp_int(x), warp_int(y)


def get_opposite(x, y) -> tuple:
    # Returns the mirrored point across the diagonal
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def in_bounds(x, y) -> bool:
    return 0 <= x < Global.SPACE_SIZE and 0 <= y < Global.SPACE_SIZE


def is_upper_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id, x, y) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)


def will_shift_happen(step, speed) -> bool:
    step -= 1
    return (step - 1) * speed % 1 > step * speed % 1


def is_obstacle_shift_possible(step) -> bool:
    return any(will_shift_happen(step, Global.POSSIBLE_OBSTACLE_MOVEMENT_SPEED))


def is_obstacles_shifted(step) -> bool:
    return all(will_shift_happen(step, Global.POSSIBLE_OBSTACLE_MOVEMENT_SPEED))


def is_speed_possible(steps: dict, speed):
    return all(
        [
            will_shift_happen(step, speed) == shift_happened
            for step, shift_happened in steps.items()
        ]
    )


def get_circle(x, y, rad):
    points = []
    for i in range(rad):
        points.append((x - rad + i, y + i))
        points.append((x + rad - i, y - i))
        points.append((x + i, y + rad - i))
        points.append((x - i, y - rad + i))
    return points


def manhattan_distance(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def sq_distance(a: tuple, b: tuple) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def get_conditional_variants(
    variants: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    mask = np.zeros(shape=(Global.SPACE_SIZE, Global.SPACE_SIZE), dtype=np.int8)
    for result in Global.REWARD_RESULTS:
        if len(result["nodes"]) > 3:
            continue
        for node in result["nodes"]:
            x, y = node.coordinates
            mask[x, y] = 1
    conditional_variants = []
    for variant in variants:
        conditional_variant = []
        for x, y in variant:
            if mask[x, y] != 0:
                conditional_variant.append((x, y))
        conditional_variants.append(conditional_variant)
    return conditional_variants


_permutations = dict(
    [
        ((2, 1), [[0, 1], [1, 0]]),
        ((3, 1), [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        ((3, 2), [[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
    ]
)


def next_permutaion(perm):
    for i in range(len(perm)):
        perm[i]["permutation"] += 1
        if len(perm[i]["nodes"]) == perm[i]["permutation"]:
            perm[i]["permutation"] = 0
            continue
        else:
            break
    return perm


def is_first_permutaion(perm):
    for el in perm:
        if el["permutation"] != 0:
            return False
    return True


def build_reward_map_of_perm(perm):
    result = np.full(
        shape=(Global.SPACE_SIZE, Global.SPACE_SIZE), fill_value=-1, dtype=np.int8
    )
    for el in perm:
        l = len(el["nodes"])
        for i, node in enumerate(el["nodes"]):
            x, y = node.coordinates
            prev_value = result[x, y]
            result[x, y] = _permutations[(len(el["nodes"]), el["reward"])][
                el["permutation"]
            ][i]
            if prev_value != -1 and prev_value != result[x, y]:
                return None
    return result


def count_sum(map, variant):
    sum = 0
    for x, y in variant:
        if map[x, y] == -1:
            raise Exception("variant contains previously unexplored coordinates")
        sum += map[x, y]
    return sum


def conditional_prob_distrubution(
    variants_to_explore: list[list[tuple[int, int]]],
) -> list[list[float]]:
    reward_results = copy.deepcopy(Global.REWARD_RESULTS)
    reward_results = [result for result in reward_results if len(result["nodes"]) <= 3]
    printDebug("used data len: " + str(len(reward_results)))
    prob_results = [[0] * (len(variant) + 1) for variant in variants_to_explore]

    for result in reward_results:
        result["permutation"] = 0

    reward_map = build_reward_map_of_perm(reward_results)
    if reward_map is not None:
        for i in range(len(variants_to_explore)):
            prob_results[i][count_sum(reward_map, variants_to_explore[i])] += 1
    reward_results = next_permutaion(reward_results)
    while not is_first_permutaion(reward_results):
        reward_map = build_reward_map_of_perm(reward_results)
        if reward_map is not None:
            for i in range(len(variants_to_explore)):
                prob_results[i][count_sum(reward_map, variants_to_explore[i])] += 1
        reward_results = next_permutaion(reward_results)

    for i in range(len(prob_results)):
        sum = 0
        for prob in prob_results[i]:
            sum += prob
        for j in range(len(prob_results[i])):
            prob_results[i][j] /= sum
    return prob_results


def unconditional_prob_distrubution(n: int) -> list[float]:
    prob_dist = [None] * (n + 1)
    for i in range(n + 1):
        prob_dist[i] = 0.2**n * 4 ** (n - i) * math.comb(n, i)
    return prob_dist


def calc_entropy(cond_dist, uncond_dist):
    general_dist = [0] * (len(cond_dist) + len(uncond_dist) - 1)
    for i in range(len(cond_dist)):
        for j in range(len(uncond_dist)):
            general_dist[i + j] += cond_dist[i] * uncond_dist[j]
    entropy = 0
    for prob in general_dist:
        if prob == 0:
            continue
        entropy += -prob * math.log2(prob)
    return entropy
