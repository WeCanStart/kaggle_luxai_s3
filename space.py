import numpy as np
from scipy.signal import convolve2d
import copy

from node import Node
from base import (
    Global,
    NodeType,
    SPACE_SIZE,
    get_opposite,
    is_team_sector,
    is_speed_possible,
    is_obstacle_shift_possible,
    is_obstacles_shifted,
    get_match_step,
    warp_point,
    sq_distance,
    printDebug,
)
from energy_prediction import EnergyPrediction


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

        self.sure_for_energies = False
        self.energy_prediction = EnergyPrediction()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def calc_probability_of_relic_spawn(self, match_num: int) -> float:
        if match_num == 0:
            return 1
        start_prob = max(0, (3 - match_num) / 3)
        explored_part = 0
        for node in self:
            if is_team_sector(0, *node.coordinates):
                explored_part += min(node.last_explored, 50)
        explored_part /= 15000
        found_prob = start_prob * explored_part

        probability = (start_prob - found_prob) / (1 - found_prob)

        return probability

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def will_change_after_shift(self, x, y) -> bool:
        node = self.get_node(x, y)
        if node.is_unknown:
            return False
        if Global.OBSTACLE_MOVEMENT_DIRECTION != (-1, 1):
            high_neigh = self.get_node((x - 1) % 24, (y + 1) % 24)
            if high_neigh.is_unknown or high_neigh.type == node.type:
                return False
        if Global.OBSTACLE_MOVEMENT_DIRECTION != (1, -1):
            low_neigh = self.get_node((x + 1) % 24, (y - 1) % 24)
            if low_neigh.is_unknown or low_neigh.type == node.type:
                return False
        return True

    def update(self, step, obs, team_id, team_reward):
        self._update_map(obs, team_id, step)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_relic_map(self, step, obs, team_id, team_reward):
        match_step = get_match_step(step)
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                self._update_relic_status(match_step, *xy, status=True)

        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.relic:
                self._update_relic_status(match_step, *node.coordinates, status=False)
                node.last_explored = match_step

            if not node.explored_for_reward:
                all_rewards_found = False

        if all_rewards_found and not Global.ALL_REWARDS_FOUND:
            printDebug("Rewards explored")
        if not all_rewards_found and Global.ALL_REWARDS_FOUND:
            printDebug("Need to find rewards")
        Global.ALL_REWARDS_FOUND = all_rewards_found

        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) == Global.MAX_RELIC_NODES:
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.relic:
                        self._update_relic_status(
                            match_step, *node.coordinates, status=False
                        )

        if not Global.ALL_REWARDS_FOUND:
            self._update_reward_status_from_relics_distribution()
            self._update_reward_results(obs, team_id, team_reward)
            self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        # We will use Global.REWARD_RESULTS to identify which nodes yield points

        non_empty = len(Global.REWARD_RESULTS) != 0
        new_discovery = True
        while new_discovery:
            new_results = []
            new_discovery = False

            for result in Global.REWARD_RESULTS:
                new_nodes = set()
                reward_nodes = set()
                for n in result["nodes"]:
                    if n.explored_for_reward:
                        if n.reward:
                            reward_nodes.add(n)
                    else:
                        new_nodes.add(n)
                result["nodes"] = new_nodes
                result["reward"] -= len(reward_nodes)

            for result in Global.REWARD_RESULTS:
                unknown_nodes = set(result["nodes"])
                reward = result["reward"]

                if reward == 0:
                    # all nodes are empty
                    new_discovery = True
                    for node in unknown_nodes:
                        self._update_reward_status(*node.coordinates, status=False)

                elif reward == len(unknown_nodes):
                    new_discovery = True
                    # all nodes yield points
                    for node in unknown_nodes:
                        self._update_reward_status(*node.coordinates, status=True)

                elif reward > len(unknown_nodes):
                    # we shouldn't be here
                    Global.ALL_RELICS_FOUND = False
                    Global.MIN_RELIC_NODES = len(self.relic_nodes) + 2

                    printDebug(
                        f"Something wrong with reward result: {result}"
                        + ", this result will be ignored."
                    )
                else:
                    if result not in new_results:
                        new_results.append(result)
            Global.REWARD_RESULTS = new_results

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        reward_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                node = self.get_node(*position)
                if not node.explored_for_reward:
                    ship_nodes.add(node)
                if node.reward:
                    reward_nodes.add(node.coordinates)
        if len(ship_nodes) != 0:
            Global.REWARD_RESULTS.append(
                {"nodes": ship_nodes, "reward": team_reward - len(reward_nodes)}
            )

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or node.last_explored < 50:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_relic_status(self, match_step, x, y, status=True):
        node = self.get_node(x, y)
        node.update_relic_status(match_step, status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(match_step, status)

        if status:
            new_results = []
            for result in Global.REWARD_RESULTS:
                has_intersection = False
                for res_node in result["nodes"]:
                    if sq_distance(
                        res_node.coordinates, node.coordinates
                    ) <= 2 or sq_distance(res_node.coordinates, opp_node.coordinates):
                        has_intersection = True
                        break
                if not has_intersection:
                    new_results.append(result)
            Global.REWARD_RESULTS = new_results
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if not (0 <= x + dx < 24) or not (0 <= y + dy < 24):
                        continue
                    r_node = self.get_node(x + dx, y + dy)
                    opp_r_node = self.get_node(*get_opposite(x + dx, y + dy))
                    if not r_node.reward:
                        r_node._explored_for_reward = False
                    if not opp_r_node.reward:
                        opp_r_node._explored_for_reward = False

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)
        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs, team_id, step):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = None
        if is_obstacles_shifted(step):
            obstacles_shifted = True
        elif is_obstacle_shift_possible(step):
            for node in self:
                x, y = node.coordinates
                if sensor_mask[x, y] and self.will_change_after_shift(x, y):
                    obstacles_shifted = node.type.value != obs_tile_type[x, y]
                    break

            if obstacles_shifted is not None:
                Global.OBSTACLES_MOVEMENT_STATUS[step] = obstacles_shifted
                Global.POSSIBLE_OBSTACLE_MOVEMENT_SPEED = (
                    self._find_OBSTACLE_MOVEMENT_SPEED(Global.OBSTACLES_MOVEMENT_STATUS)
                )
                printDebug(Global.POSSIBLE_OBSTACLE_MOVEMENT_SPEED)
                # printDebug( f"step={step} " + str(Global.POSSIBLE_OBSTACLE_MOVEMENT_SPEED))
        else:
            obstacles_shifted = False

        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction

        if (
            not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted
        ) or obstacles_shifted is None:
            for node in self:
                node.type = NodeType.unknown

        if obstacles_shifted and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND:
            self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

        energy_nodes_shifted = False
        for node in self:
            node.banned=False
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]
            if (
                is_visible
                and node.raw_energy is not None
                and node.raw_energy != obs_energy[x, y]
            ):
                self.energy_nodes_shifted = True
                self.sure_for_energies = False

        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        def should_be_visible(node: Node, ship_nodes: set[Node]):
            visible = False
            for ship in ship_nodes:
                if (
                    sq_distance(node.coordinates, ship.coordinates)
                    <= Global.UNIT_SENSOR_RANGE
                ):
                    visible = True
                    break
            return visible

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.raw_energy = int(obs_energy[x, y])
            elif energy_nodes_shifted:
                node.raw_energy = None

            if (
                not is_visible
                and should_be_visible(node, ship_nodes)
                and not is_obstacle_shift_possible(step)
            ):
                node.type = NodeType.nebula

        if not self.sure_for_energies:
            coordinates = []
            energies = []
            for node in self:
                if node.is_visible:
                    coordinates.append(node.coordinates)
                    energies.append(node.raw_energy)
            predicted_energies, self.sure_for_energies = (
                self.energy_prediction.find_possible_energies(coordinates, energies)
            )
            for node, energy in zip(
                np.array(self._nodes).ravel(), predicted_energies.T.ravel()
            ):
                node.energy = int(energy)

    @staticmethod
    def _find_OBSTACLE_MOVEMENT_SPEED(obstacles_movement_status):
        return np.array(
            [
                speed
                for speed in Global.POSSIBLE_OBSTACLE_MOVEMENT_SPEED
                if is_speed_possible(obstacles_movement_status, speed)
            ]
        )

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False
            node.last_explored = 0

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def is_explored_before(self, coords):
        nodes = set([self.get_node(*coord) for coord in coords])
        return any([nodes == result["nodes"] for result in Global.REWARD_RESULTS])

    def print_relic_status(self, relic_node):
        minx = max(0, relic_node.coordinates[0] - 2)
        maxx = min(23, relic_node.coordinates[0] + 2)
        miny = max(0, relic_node.coordinates[1] - 2)
        maxy = min(23, relic_node.coordinates[1] + 2)
        str_result = "\n" + str(relic_node.coordinates)
        for y in range(miny, maxy + 1):
            str_result += "\n"
            for x in range(minx, maxx + 1):
                node = self.get_node(x, y)
                if node.explored_for_reward:
                    str_result += "X" if node.reward else "O"
                else:
                    str_result += "_"
        printDebug(str_result)
