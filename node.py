from base import NodeType, Global


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.is_visible = False
        self.raw_energy = None
        self.energy: int = 1
        self.banned : bool = False

        self._relic = False
        self._reward = False
        self._explored_for_reward = True
        self.last_explored = 0

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, match_step: int, status: bool = False):
        self._relic = status
        self.last_explored = match_step

    def update_reward_status(self, status: bool | None = None):
        # add case with reward change

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid and not self.banned

    @property
    def is_rewardable(self) -> bool:
        energy = self.energy
        if self.type == NodeType.nebula:
            energy -= Global.NEBULA_ENERGY_REDUCTION
        return not (self.type == NodeType.asteroid or energy <= -10)

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y
    
    @property
    def full_energy(self) -> int:
        energy = self.energy
        if self.type == NodeType.nebula:
            energy -= Global.NEBULA_ENERGY_REDUCTION
        return energy