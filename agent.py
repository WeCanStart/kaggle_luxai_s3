import os
import copy
import numpy as np
import math

from fleet import Fleet
from space import Space
from node import Node
from ship import Ship

from base import (
    Global,
    NodeType,
    ActionType,
    printDebug,
    get_match_step,
    get_match,
    is_team_sector,
    get_circle,
    in_bounds,
    is_obstacles_shifted,
    sq_distance,
    get_opposite
)
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
    Weights,
)


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        Global.TEAM_ID = self.team_id
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.memo_enemy_pos_value = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
        self.memo_enemy_move_dist = dict()
        # printDebug(env_cfg)

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.next_opposite_fleet = np.zeros(
            (Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32
        )

        self.space = Space()
        self.prev_fleet = Fleet(self.team_id)
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

        self.weights = Weights()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        Global.STEP = step

        # if ((step % 30 == 0 and 0<step<202) or step == 102) and self.team_id==0:
        #     printDebug("")
        #     show_map(self.space, self.fleet, only_visible=False)


        if match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.update(step, obs, self.team_id, 0)
            return self.create_actions_array()
        self.fleet.clear_low()
        our_ships_value = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
            

        self.memo_enemy_move_dist=dict()

        points = int(obs["team_points"][self.team_id])
        Global.POINTS_GAIN = obs["team_points"][self.team_id] - Global.POINTS
        Global.POINTS = obs["team_points"][self.team_id]
        Global.ENEMY_POINTS_GAIN = (
            obs["team_points"][self.opp_team_id] - Global.ENEMY_POINTS
        )
        Global.ENEMY_POINTS = obs["team_points"][self.opp_team_id]

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)
        for ship in self.fleet:
            for opp_ship in self.opp_fleet:
                if opp_ship.energy+10>=ship.energy:
                    if manhattan_distance(ship.coordinates,opp_ship.coordinates)==1:
                        opp_ship.node.banned=True
                    if manhattan_distance(ship.coordinates,opp_ship.coordinates)==2:
                        for pos in get_circle(*opp_ship.coordinates,1):
                            if manhattan_distance(ship.coordinates,pos)==1:
                                self.space.get_node(*pos).banned=True
                else:
                    if manhattan_distance(ship.coordinates,opp_ship.coordinates)==1 and opp_ship.node.is_walkable and self.opp_fleet.count_ships(opp_ship.node)==1:
                        ship.task="killer"
                        ship.action=ActionType.from_coordinates(ship.coordinates, opp_ship.coordinates)
        self.do_dirty(obs)

        for ship in self.fleet:
            ship_value=1
            if not ship.node.reward:
                ship_value*=0.4
            our_ships_value[ship.coordinates]+=ship_value*(1-Global.DROPOFF)
            for pos in nearby_positions(*ship.coordinates,1):
                our_ships_value[pos]+=ship_value*Global.DROPOFF
        for x in range(24):
            for y in range(24):
                x_min=max(0,x-Global.UNIT_SAP_RANGE)
                x_max=min(23,x+Global.UNIT_SAP_RANGE)+1
                y_min=max(0,y-Global.UNIT_SAP_RANGE)
                y_max=min(23,y+Global.UNIT_SAP_RANGE)+1
                self.memo_enemy_pos_value[(x,y)]=np.max(our_ships_value[x_min:x_max,y_min:y_max])

        self.next_opposite_fleet = self.calc_reward_enemy_distribution()

        for ship in self.opp_fleet:
            if ship.energy < 0:
                continue
            dist = self.get_ship_move_dist(ship)
            for pos, prob in dist.items():
                weighted_prob=prob
                if ship.energy>Global.UNIT_SAP_COST-10:
                    weighted_prob*=max(1,self.evaluate_enemy_pos(*pos))
                else:
                    if ship.node.reward:
                        weighted_prob*=2
                    else:
                        weighted_prob*=0.5
                self.next_opposite_fleet[pos]+=weighted_prob


        self.harvest()
        self.find_relics()
        self.find_rewards()
        self.contest()
        self.gain_energy()
        self.shoot()
        # printDebug(self.profiler.output_text(unicode=True, color=True, show_all=True))

        return self.create_actions_array()
    

    def evaluate_enemy_pos(self,x,y):
        return self.memo_enemy_pos_value[(x,y)]
    

    def get_ship_move_dist(self,opp_ship : Ship):
        if not opp_ship.unit_id in self.memo_enemy_move_dist:
            dist= dict()
            c = 1
            enemy_pos_value=self.evaluate_enemy_pos(*opp_ship.coordinates)
            if opp_ship.node.reward:
                dist[opp_ship.coordinates] = 0.75
            elif enemy_pos_value<1:
                dist[opp_ship.coordinates] = 0.1
            else:
                dist[opp_ship.coordinates] = min(0.2*enemy_pos_value,0.9)
            c -= dist[opp_ship.coordinates]
            ens = dict()
            sum = 0
            for x, y in get_circle(*opp_ship.coordinates, 1):
                if in_bounds(x, y) and self.space.get_node(x, y).is_walkable:
                    ens[(x, y)] = self.space.energy_prediction.last_prediction[x, y]
                    if self.space.get_node(x, y).type == NodeType.nebula:
                        ens[(x, y)] -= Global.NEBULA_ENERGY_REDUCTION
                    ens[(x, y)] = 1.1 ** (ens[(x, y)])
                    sum += ens[(x, y)]

            for coords, energy in ens.items():
                dist[coords] = c * energy / sum
            self.memo_enemy_move_dist[opp_ship.unit_id]=dist
        return self.memo_enemy_move_dist[opp_ship.unit_id]

    def do_dirty(self, obs):
        for prev_ship, ship in zip(
            self.fleet.prev_ships, self.fleet.ships
        ):  # BEEEEEEEE CAAAAAAAAAAAAAREEEEEEEEEEFUUUUUUUUUUL no more ship.energy is not None
            if (
                (prev_ship.node is not None)
                and (ship.node is not None)
                and ship.node.type == NodeType.nebula
                and prev_ship.energy >= 1
                and ship.energy >= 1
                and sq_distance(self.fleet.spawn_point, ship.coordinates)
                < Global.SPACE_SIZE // 2
            ):
                energy_drop = (
                    prev_ship.energy
                    - ship.energy
                    + ship.node.energy
                    - Global.UNIT_MOVE_COST
                    * int(prev_ship.coordinates != ship.coordinates)
                    - Global.UNIT_SAP_COST * int(prev_ship.action == ActionType.sap)
                )
                for i in range(len(Global.NEBULAS_MEET)):
                    if Global.NEBULAS_POSSIBLE[i] == energy_drop:
                        Global.NEBULAS_MEET[i] += (
                            1 if (Global.NEBULAS_POSSIBLE[i] != 0) else 0.1
                        )
                        if Global.NEBULA_ENERGY_REDUCTION != Global.NEBULAS_POSSIBLE[
                            i
                        ] and Global.NEBULAS_MEET[i] == max(Global.NEBULAS_MEET):
                            Global.NEBULA_ENERGY_REDUCTION = Global.NEBULAS_POSSIBLE[i]
                            printDebug("energy drop by nebula: " + str(Global.NEBULA_ENERGY_REDUCTION))

        sensor_mask = obs["sensor_mask"]

        view_map = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.int32)
        for ship in self.fleet:
            for coordinates in nearby_positions(
                *ship.node.coordinates, Global.UNIT_SENSOR_RANGE
            ):
                # printDebug(coordinates)
                if coordinates == ship.node.coordinates:
                    view_map[coordinates] += 10
                else:
                    view_map[coordinates] += (
                        Global.UNIT_SENSOR_RANGE
                        + 1
                        - sq_distance(ship.node.coordinates, coordinates)
                    )
        # if Global.STEP == 30 and self.team_id == 0:
        #     printDebug("------------")
        #     printDebug(view_map)
        #     printDebug("------------")
        #     printDebug(sensor_mask.astype(int))
        #     printDebug("------------")

        for node in self.space:
            # if bool(sensor_mask[*node.coordinates]) and NodeType(int(obs_tile_type[*node.coordinates])) == NodeType.nebula: #тупые разрабы
            #     if Global.UPPER_NEBULA_RANGE_REDUCTION > view_map[*node.coordinates]:
            #         Global.UPPER_NEBULA_RANGE_REDUCTION = view_map[*node.coordinates]
            #         if self.team_id == 0:
            #             printDebug("range " + str(Global.LOWER_NEBULA_RANGE_REDUCTION + 1) + " " + str(Global.UPPER_NEBULA_RANGE_REDUCTION))
            if (
                not bool(sensor_mask[node.coordinates])
                and view_map[node.coordinates] != 0
            ):
                if Global.LOWER_NEBULA_RANGE_REDUCTION < view_map[node.coordinates]:
                    Global.LOWER_NEBULA_RANGE_REDUCTION = view_map[node.coordinates]
                    # if self.team_id == 0:
                    #     printDebug((Global.LOWER_NEBULA_RANGE_REDUCTION + 1 + 8) // 2)

        for i, (prev_opp_ship, opp_ship) in enumerate(
            zip(self.opp_fleet.prev_ships, self.opp_fleet.ships)
        ):
            close = 0
            en_diff = 0
            if (
                (prev_opp_ship.node is not None)
                and (opp_ship.node is not None)
                and prev_opp_ship.energy >= 1
            ):
                en_diff = (
                    prev_opp_ship.energy
                    - opp_ship.energy
                    + opp_ship.node.energy
                    - Global.UNIT_MOVE_COST
                    * int(prev_opp_ship.coordinates != opp_ship.coordinates)
                    - Global.NEBULA_ENERGY_REDUCTION
                    * int(opp_ship.node.type == NodeType.nebula)
                )
                stands_too_close = False
                for prev_ship, ship in zip(self.fleet.prev_ships, self.fleet.ships):
                    if (
                        (prev_ship.node is not None)
                        and (ship.node is not None)
                        and manhattan_distance(ship.coordinates, opp_ship.coordinates)
                        <= 1
                    ):
                        stands_too_close = True
                        break
                    if (
                        (prev_ship.node is not None)
                        and (ship.node is not None)
                        and prev_ship.action == ActionType.sap
                    ):
                        if prev_ship.sap_target == opp_ship.node:
                            en_diff -= Global.UNIT_SAP_COST
                        elif (
                            sq_distance(
                                prev_ship.sap_target.coordinates, opp_ship.coordinates
                            )
                            == 1
                        ):
                            close += 1
                if stands_too_close:
                    continue
            if en_diff == 2 * Global.UNIT_SAP_COST:
                if close == 2:
                    Global.DROPOFFS_MEET[0] -= 1
                    Global.DROPOFFS_MEET[1] += 1
                    Global.DROPOFFS_MEET[2] += 1
                    for df_cnt in range(len(Global.DROPOFFS_MEET)):
                        if Global.DROPOFF != Global.DROPOFFS_POSSIBLE[
                            df_cnt
                        ] and Global.DROPOFFS_MEET[df_cnt] == max(Global.DROPOFFS_MEET):
                            Global.DROPOFF = Global.DROPOFFS_POSSIBLE[df_cnt]
                            printDebug("DROPOFF " + str(Global.DROPOFF))
                    continue
                elif close == 4:
                    Global.DROPOFFS_MEET[0] += 1
                    Global.DROPOFFS_MEET[1] += 1
                    Global.DROPOFFS_MEET[2] -= 1
                    for df_cnt in range(len(Global.DROPOFFS_MEET)):
                        if Global.DROPOFF != Global.DROPOFFS_POSSIBLE[
                            df_cnt
                        ] and Global.DROPOFFS_MEET[df_cnt] == max(Global.DROPOFFS_MEET):
                            Global.DROPOFF = Global.DROPOFFS_POSSIBLE[df_cnt]
                            printDebug(
                                "DROPOFF "
                                + str(Global.DROPOFF)
                                + " "
                                + str(en_diff)
                                + " "
                                + str(close)
                            )
                    continue
            if close != 0:
                val1 = en_diff
                val2 = en_diff - Global.UNIT_SAP_COST
                #printDebug(
                #    "Possible dropoff " + str(val1) + " " + str(val2) + " " + str(close)
                #)
                val = -1
                if val1 in (
                    Global.DROPOFFS_POSSIBLE * Global.UNIT_SAP_COST * close
                ).astype(int):
                    val = val1
                elif val2 in (
                    Global.DROPOFFS_POSSIBLE * Global.UNIT_SAP_COST * close
                ).astype(int):
                    val = val2
                for df_cnt in range(len(Global.DROPOFFS_MEET)):
                    if (
                        int(
                            Global.DROPOFFS_POSSIBLE[df_cnt]
                            * Global.UNIT_SAP_COST
                            * close
                        )
                        == val
                    ):
                        Global.DROPOFFS_MEET[df_cnt] += 1
                        if Global.DROPOFF != Global.DROPOFFS_POSSIBLE[
                            df_cnt
                        ] and Global.DROPOFFS_MEET[df_cnt] == max(Global.DROPOFFS_MEET):
                            Global.DROPOFF = Global.DROPOFFS_POSSIBLE[df_cnt]
                            printDebug(
                                "DROPOFF "
                                + str(Global.DROPOFF)
                                + " "
                                + str(en_diff)
                                + " "
                                + str(close)
                            )

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action == ActionType.sap:
                # print(ship.target, ship.target.x, ship.target.y, file=stderr)
                actions[i] = (
                    ship.action,
                    ship.sap_target.x - ship.coordinates[0],
                    ship.sap_target.y - ship.coordinates[1],
                )
            elif ship.action is not None:
                actions[i] = ship.action, 0, 0

        return actions

    def calc_reward_enemy_distribution(self, conditional : bool =True):


        unknown_rewards = Global.ENEMY_POINTS_GAIN
        for ship in self.opp_fleet:
            if ship.energy < 0:
                continue
            if ship.node.reward:
                unknown_rewards -= 1
            if not ship.node.explored_for_reward:
                unknown_rewards -= 0.2
        distribution = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)

        reward_nodes = []
        max_val = 0
        min_val = 10000000000000000
        avg_val = 0
        count = 0

        for node in self.space:
            if (node.explored_for_reward and not node.reward) or (node.is_visible and conditional) or (not conditional and not is_team_sector(self.opp_fleet.team_id,*node.coordinates)):
                continue

            dens = node.energy
            if node.type == NodeType.nebula:
                dens -= Global.NEBULA_ENERGY_REDUCTION
            dens = 1.05**dens
            dens *= 0.9 ** manhattan_distance(
                node.coordinates, self.opp_fleet.spawn_point
            )
            if not node.is_walkable:
                dens *= 0.75
            if not node.explored_for_reward:
                dens *= 0.2
            if not is_team_sector(self.opp_fleet.team_id,*node.coordinates):
                dens *=0.375
            distribution[node.coordinates] = dens
            reward_nodes.append(node.coordinates)
            count += 1
            max_val = max(max_val, dens)
            min_val = min(min_val, dens)
            avg_val += dens
        
        if not conditional:
            unknown_rewards=count/2

        if count != 0 and count != 1:
            avg_val /= count
            if min_val != max_val:
                c = (
                    min(
                        (unknown_rewards) / (avg_val - min_val),
                        (count - unknown_rewards) / (max_val - avg_val),
                    )
                    * 0.9
                )
                for node in reward_nodes:
                    distribution[node] = (
                        unknown_rewards / count
                        + c * (distribution[node] - avg_val) / count
                    )
                    distribution[node] = float(distribution[node])
            else:
                for node in reward_nodes:
                    distribution[node] = unknown_rewards / count
                    distribution[node] = float(distribution[node])
        if len(distribution) == unknown_rewards:
            for node in reward_nodes:
                distribution[node] = 1

        return distribution

    def find_relics(self):
        count_relic_nodes=0
        for relic in self.space.relic_nodes:
            if relic.coordinates == get_opposite(*relic.coordinates):
                count_relic_nodes+=1
            else:
                count_relic_nodes+=0.5
        if (
            Global.ALL_RELICS_FOUND
            or (get_match(Global.STEP) + 1) == count_relic_nodes
            or (get_match(Global.STEP) + 1) >= count_relic_nodes + 2
        ):
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return
        elif (
            Global.MIN_RELIC_NODES <= len(self.space.relic_nodes)
        ) and self.space.calc_probability_of_relic_spawn(get_match(Global.STEP)) < 0.05:
            Global.ALL_RELICS_FOUND = True
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        def evaluate_vision_from(x, y):
            node = self.space.get_node(x, y)
            if node.type == NodeType.asteroid:
                return -1
            evaluation = 0
            for dx in range(-Global.UNIT_SENSOR_RANGE, Global.UNIT_SENSOR_RANGE + 1):
                for dy in range(
                    -Global.UNIT_SENSOR_RANGE, Global.UNIT_SENSOR_RANGE + 1
                ):
                    if (
                        not (0 <= x + dx < 24)
                        or not (0 <= y + dy < 24)
                        or not is_team_sector(self.fleet.team_id, x + dx, y + dy)
                    ):
                        continue
                    if (
                        (dx != 0 or dy != 0)
                        and self.space.get_node(x + dx, y + dy).type == NodeType.nebula
                        and Global.LOWER_NEBULA_RANGE_REDUCTION + max(abs(dx), abs(dy))
                        > Global.UNIT_SENSOR_RANGE
                    ):
                        continue
                    evaluation += max(
                        0,
                        min(Global.STEP, 50)
                        - self.space.get_node(x + dx, y + dy).last_explored,
                    )
            return evaluation

        targets = set()

        def set_task(ship):
            if ship.energy < Global.UNIT_MOVE_COST * 5 + 5:
                return False
            if ship.task and ship.task != "find_relics":
                return False
            if (
                ship.task is None
                or ship.target == ship.coordinates
                or is_obstacles_shifted(Global.STEP)
            ):
                best_evaluation = 0
                best_action = ActionType.center
                target = ship.target
                for rad in range(1, 10):
                    if ship.energy < rad * Global.UNIT_MOVE_COST + rad:
                        break
                    for x, y in get_circle(*ship.coordinates, rad):
                        if not (0 <= x < 24) or not (0 <= y < 24) or (x, y) in targets:
                            continue
                        evaluation = evaluate_vision_from(x, y)
                        path = astar(
                            self.weights.create_weights(self.space),
                            ship.coordinates,
                            (x, y),
                        )
                        if len(path) == 0:
                            continue
                        evaluation -= Global.UNIT_SENSOR_RANGE * estimate_energy_cost(
                            self.space, path
                        )
                        actions = path_to_actions(path)
                        if len(actions) == 0:
                            continue
                        evaluation /= math.sqrt(len(actions))
                        if evaluation > best_evaluation:
                            best_evaluation = evaluation
                            target = (x, y)
                            best_action = actions[0]

                if best_evaluation == 0:
                    return False
                ship.task = "find_relics"
                ship.target = target
                ship.action = best_action
            else:
                path = astar(
                    self.weights.create_weights(self.space),
                    ship.coordinates,
                    ship.target,
                )
                if len(path) == 0:
                    return False
                actions = path_to_actions(path)
                ship.action = actions[0]
            targets.add(ship.target)
            return True


        max_amount = 30 // (Global.UNIT_SENSOR_RANGE)
        if len(self.space.reward_nodes)>0:
            max_amount=2
        count = 0
        for ship in self.fleet:
            if ship.task == "find_relics" and set_task(ship):
                count += 1
            elif ship.task == "find_relics":
                ship.task = None
                ship.target = None

        for ship in self.fleet:
            if not ship.task and count != max_amount :
                if set_task(ship):
                    count += 1

    def find_wayouts(self, node: Node):
        return [
            way
            for way in get_circle(*node.coordinates, 1)
            if in_bounds(*way)
            and not self.space.get_node(*way).explored_for_reward
            and self.space.get_node(*way).is_rewardable
        ]

    def find_wayins(self, node: Node):
        if not node.is_rewardable:
            return []
        return [
            ship.coordinates
            for ship in self.fleet
            if manhattan_distance(ship.coordinates, node.coordinates) == 1
            and ship.task == "find_rewards"
        ]

    def get_explorable_rewards(self, relic: Node):
        base = []
        for x, y in nearby_positions(*relic.coordinates, Global.RELIC_REWARD_RANGE):
            node = self.space.get_node(x, y)
            if node.is_rewardable and not node.explored_for_reward:
                base.append(node)
        changed = True
        result = copy.deepcopy(base)
        while changed:
            changed = False
            new_result = []
            for node in result:
                wayouts = len(self.find_wayouts(node))
                if len(self.find_wayins(node)) > 0:
                    wayouts += 1
                if wayouts <= 1:
                    changed = True
                else:
                    new_result.append(node)
            if changed:
                result = new_result
        if len(result) != 0 and len(base) / len(result) < 3:
            return result
        else:
            return base

    def get_options(self, ship, relic):
        relic_options = self.get_explorable_rewards(relic)
        ship_options = [
            self.space.get_node(x, y)
            for x, y in get_circle(*ship.coordinates, 1)
            if in_bounds(x, y)
        ]
        return list(set(relic_options) & set(ship_options))

    def get_best_options(self, options: list[Node]) -> list[Node]:
        min_wayouts = 5
        best_options = []
        for option in options:
            wayouts = len(self.find_wayouts(option))
            if wayouts == min_wayouts:
                best_options.append(option)
            if wayouts < min_wayouts:
                best_options = [option]
                min_wayouts = wayouts
        return best_options

    def get_unexplored_rewards(self, relic=None) -> list[Node]:
        reward_nodes = []
        for node in self.space:
            if (
                is_team_sector(self.team_id, *node.coordinates)
                and not node.explored_for_reward
                and (relic is None or sq_distance(relic.coordinates, node.coordinates))
            ):
                reward_nodes.append(node.coordinates)
        return reward_nodes

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)
        return relic_nodes

    def find_rewards(self):
        unexplored_rewards = self.get_unexplored_rewards()

        if len(unexplored_rewards) == 0 and not Global.ALL_REWARDS_FOUND:
            printDebug("Rewards explored")
            Global.ALL_REWARDS_FOUND = True
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        if all(
            [
                not self.space.get_node(*node).is_rewardable
                for node in unexplored_rewards
            ]
        ):
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        top_2 = []
        unexplored_relics = self.get_unexplored_relics()
        # for relic in unexplored_relics:
        #     self.space.print_relic_status(relic)
        interesting_relics = [
            relic
            for relic in unexplored_relics
            if len(self.get_explorable_rewards(relic)) != 0
        ]

        if len(interesting_relics) == 0:
            return
        if len(interesting_relics) == 1:
            top_2 = [unexplored_relics[0], unexplored_relics[0]]
        else:
            count_rewards = sorted(
                [
                    (len(self.get_explorable_rewards(relic)), i)
                    for i, relic in enumerate(interesting_relics)
                ],
                reverse=True,
            )
            top_2 = [
                unexplored_relics[count_rewards[0][1]],
                unexplored_relics[count_rewards[1][1]],
            ]

        ship_to_relic_node = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.attached_relic in unexplored_relics
                    and ship.energy
                    > Global.UNIT_MOVE_COST * 5 + 5  # в бейзлайне тут было * 5
                ):
                    ship_to_relic_node[ship] = ship.attached_relic
                else:
                    ship.task = None
                    ship.target = None

        for relic in top_2:
            if (
                relic not in ship_to_relic_node.values() or top_2[0] == top_2[1]
            ) and len(ship_to_relic_node) < min(2, len(unexplored_rewards)):
                count = 0
                min_distance, closest_ship = float("inf"), None
                for ship in self.fleet:
                    if (
                        (ship.task and ship.task != "find_rewards")
                        or (ship.energy < Global.UNIT_MOVE_COST * 5 + 5)
                        or (ship in ship_to_relic_node)
                    ):
                        continue
                    count += 1

                    distance = sq_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closest_ship = distance, ship

                if closest_ship:
                    ship_to_relic_node[closest_ship] = relic
                    closest_ship.attached_relic = relic

        def set_task(ship, target):
            if not target:
                return
            path = astar(
                self.weights.create_weights(self.space), ship.coordinates, target
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]
                return True

        targets = {}
        stands_on_unexplored_count = 0

        for s in ship_to_relic_node:
            if not self.space.get_node(*s.coordinates).explored_for_reward:
                stands_on_unexplored_count += 1

        ship_list = sorted(
            list(ship_to_relic_node.items()),
            key=lambda _: (
                len(self.get_options(_[0], _[1])),
                len(self.get_best_options(self.get_options(_[0], _[1]))),
            ),
        )

        for s, n in ship_list:
            if stands_on_unexplored_count == 2 and ship_list[0][0] == s:
                s.task = "find_rewards"
                s.target = s.node
                targets[s] = s.coordinates
                s.action = ActionType.center
                continue

            target = None
            options = self.get_options(s, n)
            if len(targets) != 0 and len(options) != 0:
                options = [
                    option
                    for option in options
                    if not self.space.is_explored_before(
                        list(targets.values()) + [option.coordinates]
                    )
                ]
                new_options = [
                    option
                    for option in options
                    if find_closest_target(option.coordinates, targets.values())[1] > 1
                ]
                if len(new_options) > 0:
                    options = new_options
            if len(options) == 0:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        node.coordinates
                        for node in self.get_explorable_rewards(n)
                        if node.coordinates not in targets.values()
                    ],
                )
            else:
                target = self.get_best_options(options)[0].coordinates
            targets[s] = target
            if not set_task(s, target):
                if s.task == "find_rewards":
                    s.task = None
                    s.target = None
                targets.pop(s, None)

    def harvest(self):
        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                self.weights.create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()

        for ship in self.fleet:
            if (ship.task=="find_rewards" and ship.node.reward and self.fleet.count_ships(ship.node)==1):
                if set_task(ship, ship.node):
                    booked_nodes.add(ship.node)

        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if (
                n.is_walkable
                and n not in booked_nodes
                and is_team_sector(self.fleet.team_id, *n.coordinates)
            ):
                targets.add(n.coordinates)
        if not targets:
            return
        
        free_ships = dict([(ship.coordinates,ship) for ship in self.fleet if not ship.task])

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

    def gain_energy(self):
        def set_task(ship, target_node):
            if not target:
                return False
            if ship.node == target_node:
                # printDebug("ship")
                # printDebug(ship)
                ship.task = None
                ship.target = None
                ship.action = ActionType.center
                return True
            path = astar(
                self.weights.create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "gain_energy"
            ship.target = target_node
            ship.action = actions[0]
            return True

        def evaluate_spot(ship: Ship, node: Node):
            if not node.is_walkable or not node.explored_for_reward or not is_team_sector(Global.TEAM_ID,*node.coordinates):
                return -1
            # crowd_penalty= [2 , 0.25]
            node_energy_coeff = 8
            energy_cost_coeff = 0.2
            distance_from_rewards_coeff = 2
            distance_from_spawn_coeff = 1

            metric = node.energy * node_energy_coeff

            if node.type == NodeType.nebula:
                metric -= node_energy_coeff * Global.NEBULA_ENERGY_REDUCTION
            if metric <= 0:
                return -1
            energy = (1 + Global.UNIT_MOVE_COST) * manhattan_distance(
                node.coordinates, ship.coordinates
            )
            metric -= energy * energy_cost_coeff
            if energy > ship.energy:
                return -1
            if len(self.space.relic_nodes) != 0:
                reward_distance_sum = 0
                our_relics = [relic for relic in self.space.relic_nodes if is_team_sector(Global.TEAM_ID,*relic.coordinates)]
                our_rewards = [reward for reward in self.space.reward_nodes if is_team_sector(Global.TEAM_ID,*reward.coordinates)]
                for reward in our_rewards:
                    distance = abs(manhattan_distance(node.coordinates, reward.coordinates)-(Global.UNIT_SENSOR_RANGE+1)//2)
                    reward_distance_sum += 1 / (distance + 1)
                for relic in our_relics:
                    distance = abs(manhattan_distance(node.coordinates, relic.coordinates)-(Global.UNIT_SENSOR_RANGE+1)//2)
                    reward_distance_sum += 1 / (distance + 1)
                reward_distance_sum = (1 / reward_distance_sum) * (len(our_relics) + len(our_rewards))
                metric -= reward_distance_sum * distance_from_rewards_coeff
            metric += manhattan_distance(node.coordinates,self.fleet.spawn_point)*distance_from_spawn_coeff

            metric += 1000
            return metric

        free_ships = dict(
            [
                (ship, None)
                for ship in self.fleet
                if ship.task is None and ship.energy < 250
            ]
        )
        for ship in sorted(free_ships, key=lambda _: _.energy):
            best_metric = 0
            for node in self.space:
                if node in [
                    target for target in free_ships.values() if target is not None
                ]:
                    continue
                metric = evaluate_spot(ship, node)
                if metric > best_metric:
                    best_metric = metric
                    free_ships[ship] = node

        free_ships = dict([ship for ship in free_ships.items() if ship[1] is not None])

        for ship, target in free_ships.items():
            for actual_ship in free_ships:
                if actual_ship.coordinates == target.coordinates:
                    free_ships[ship], free_ships[actual_ship] = (
                        free_ships[actual_ship],
                        free_ships[ship],
                    )
                    break
        for ship, target in free_ships.items():
            if set_task(ship, target):
                continue

            if ship.task == "gain_energy":
                ship.target = None
                ship.task = None

    def contest(self):
        targets = set()
        for node in self.space.reward_nodes:
            if not is_team_sector(self.fleet.team_id, *node.coordinates):
                targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "contest":
                return False

            if ship.energy < Global.UNIT_SAP_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False
            path = astar(
                self.weights.create_weights(self.space), ship.coordinates, target
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "contest"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]
                targets.remove(target)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "contest":
                ship.task = None
                ship.target = None

    def shoot(self):
        enemy_ships_expected_damage = dict([(ship,0) for ship in self.opp_fleet])

        def eval_shot(x, y):
            return sum(
                [self.next_opposite_fleet[cords] for cords in nearby_positions(x, y, 1)]
            ) * Global.DROPOFF + self.next_opposite_fleet[(x, y)] * (1 - Global.DROPOFF)

        for ship in self.fleet:
            # print(ship.energy, Global.UNIT_SAP_COST, Global.UNIT_SAP_RANGE, file=stderr)
            if ship.task == "killer":
                continue
            if ship.energy < Global.UNIT_SAP_COST:
                continue
            # print(ship, target, '\n', file=stderr)

            best_sap = None
            best_sap_value = 0
            treshold = 1
            for target in nearby_positions(*ship.coordinates, Global.UNIT_SAP_RANGE):
                sap_value = eval_shot(*target)
                if sap_value > best_sap_value:
                    best_sap = target
                    best_sap_value = sap_value

            if best_sap_value >= treshold:
                ship.action = ActionType.sap
                ship.sap_target = self.space.get_node(*best_sap)
                for ship in self.opp_fleet:
                    dist = self.get_ship_move_dist(ship)
                    for pos,prob in dist.items():
                        if sq_distance(pos,best_sap)==0:
                            enemy_ships_expected_damage[ship]+=prob*Global.UNIT_SAP_COST
                        if sq_distance(pos,best_sap)==1:
                            enemy_ships_expected_damage[ship]+=prob*Global.DROPOFF*Global.UNIT_SAP_COST
                    if enemy_ships_expected_damage[ship]>ship.energy:
                        for pos,prob in dist.items():
                            self.next_opposite_fleet[pos]-=prob


    def show_visible_energy_field(self):
        printDebug("Visible energy field:")
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        printDebug("Explored energy field:")
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        printDebug("Visible map:")
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        printDebug("Explored map:")
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        printDebug("Exploration map:")
        show_exploration_map(self.space)
