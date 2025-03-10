from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class PosDir:
    xpos: float
    ypos: float
    direction: float


@dataclass_json
@dataclass
class SimulationState:
    robot1: PosDir
    robot2: PosDir
