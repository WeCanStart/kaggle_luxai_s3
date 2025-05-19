from base import Global, get_match, get_match_step,sq_distance
from ship import Ship
from space import Space
import numpy as np
import copy


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.spawn_point = (0, 0) if (team_id == 0) else (23, 23)
        self.points: int = 0  # how many points have we scored in this match so far
        self.prev_ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None and not ship.is_almost_dead:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def clear_low(self):
        for ship in self.ships:
            if ship.task == "gain_energy" or ship.task == "killer" or ship.task == "contest":
                ship.clean_low()

    def update(self, obs, space: Space):
        self.prev_ships = copy.deepcopy(self.ships)
        self.points = int(obs["team_points"][self.team_id])
        # printDebug("points " + str(self.points))

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                if ship.is_almost_dead:
                    ship.clean()
                ship.is_almost_dead = False
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
            else:
                ship.clean()
        # if Global.STEP % 10 == 0:
        #     printDebug('--------------------')
        #     printDebug(f"step={Global.STEP}")
        #     printDebug('--------------------')
        #     printDebug(self.prev_ships)
        #     printDebug('--------------------')
        #     printDebug(self.ships)
        #     printDebug('--------------------')

        Global.RESERCHER_NEED = (15 if get_match(Global.STEP) == 0 else 10) * np.exp(
            -0.035 * get_match_step(Global.STEP)
        )

    def get_nearby_ships(self, x, y, radius):
        for ship in self.ships:
            if (
                ship
                and ship.node
                and sq_distance(ship.node.coordinates, (x, y)) <= radius
            ):
                yield ship
    def count_ships(self, node):
        count=0
        for ship in self:
            if ship.node==node:
                count+=1
        return count
