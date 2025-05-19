from node import Node
from base import ActionType


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None
        self.is_almost_dead: bool = False

        self.task: str | None = None
        self.target: Node | None = None
        self.sap_target: Node | None = None
        self.action: ActionType | None = None

    def __repr__(self):
        return (
            "No Ship"
            if self.node is None
            else f"Ship({self.unit_id}, task={self.task}, target={self.target}, energy={self.energy}, action={self.action})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.is_almost_dead = True if not self.is_almost_dead else False
        if not self.is_almost_dead:
            self.energy = 0
            self.node = None
            self.task = None
            self.target = None
            self.action = None
            self.sap_target = None

    def clean_low(self):
        self.task = None
        self.target = None
