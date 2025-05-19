import heapq
import numpy as np
import math

from space import Space
from base import SPACE_SIZE, NodeType, Global, ActionType, manhattan_distance,printDebug

CARDINAL_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def astar(weights: np.ndarray, start: tuple, goal: tuple) -> list:
    # A* algorithm
    # returns the shortest path form start to goal

    min_weight = weights[np.where(weights >= 0)].min()

    def heuristic(p1, p2):
        return min_weight * manhattan_distance(p1, p2)
        # return 0

    queue = []

    # nodes: [x, y, (parent.x, parent.y, distance, f)]
    nodes = np.zeros((*weights.shape, 4), dtype=np.float32)
    nodes[:] = -1

    heapq.heappush(queue, (0, start))
    nodes[start[0], start[1], :] = (*start, 0, heuristic(start, goal))

    while queue:
        f, (x, y) = heapq.heappop(queue)

        if (x, y) == goal:
            return reconstruct_path(nodes, start, goal)

        if f > nodes[x, y, 3]:
            continue

        distance = nodes[x, y, 2]
        for x_, y_ in get_neighbors(x, y):
            cost = weights[y_, x_]
            if cost < 0:
                continue

            new_distance = distance + cost
            if nodes[x_, y_, 2] < 0 or nodes[x_, y_, 2] > new_distance:
                new_f = new_distance + heuristic((x_, y_), goal)
                nodes[x_, y_, :] = x, y, new_distance, new_f
                heapq.heappush(queue, (new_f, (x_, y_)))

    return []


def get_neighbors(x: int, y: int):
    for dx, dy in CARDINAL_DIRECTIONS:
        x_ = x + dx
        if x_ < 0 or x_ >= SPACE_SIZE:
            continue

        y_ = y + dy
        if y_ < 0 or y_ >= SPACE_SIZE:
            continue

        yield x_, y_


def reconstruct_path(nodes: np.ndarray, start: tuple, goal: tuple) -> list:
    p = goal
    path = [p]
    while p != start:
        x = int(nodes[p[0], p[1], 0])
        y = int(nodes[p[0], p[1], 1])
        p = x, y
        path.append(p)
    return path[::-1]


def nearby_positions(x: int, y: int, distance: int):
    for x_ in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for y_ in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield x_, y_


def non_negative_weight(weight,min_weight):
    if min_weight>=0:
        return weight
    if weight >= 0:
        return math.sqrt(weight**2-2*min_weight*weight) - min_weight
    else:
        return -math.sqrt(2*min_weight*weight-weight**2) - min_weight


class Weights:
    def __init__(self):
        self.weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        self.step: int = -1

    def create_weights(self, space: Space) -> np.ndarray:
        # create weights for AStar algorithm

        if self.step == Global.STEP:
            return self.weights

        self.step = Global.STEP

        field = space.energy_prediction.last_prediction

        self.weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)

        maximum = np.max(field)

        for node in space:
            if node.type == NodeType.asteroid:
                weight = -1
            else:
                weight = maximum - field[node.x][node.y] + Global.UNIT_MOVE_COST
                if node.type == NodeType.nebula:
                    weight += Global.NEBULA_ENERGY_REDUCTION
            self.weights[node.y, node.x] = weight

        return self.weights


def find_closest_target(start: tuple, targets: list) -> tuple:
    target, min_distance = None, float("inf")
    for t in targets:
        d = manhattan_distance(start, t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance


def estimate_energy_cost(space: Space, path: list) -> int:
    if len(path) <= 1:
        return 0

    energy = 0
    last_position = path[0]
    for x, y in path[1:]:
        node = space.get_node(x, y)
        energy -= space.energy_prediction.last_prediction[x, y]

        if node.type == NodeType.nebula:
            energy += Global.NEBULA_ENERGY_REDUCTION

        if (x, y) != last_position:
            energy += Global.UNIT_MOVE_COST

    return energy


def path_to_actions(path: list) -> list:
    actions = []
    if not path:
        return actions

    last_position = path[0]
    for x, y in path[1:]:
        direction = ActionType.from_coordinates(last_position, (x, y))
        actions.append(direction)
        last_position = (x, y)

    return actions
