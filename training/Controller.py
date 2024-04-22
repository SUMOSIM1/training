from enum import Enum

class SectorName(Enum):
    UNDEF,
    LEFT,
    CENTER,
    RIGHT

@dataclass
class Point2:
    xpos: float
    ypos: float

@dataclass
class PosDir:
    pos: Point2
    dir: float

@dataclass
class RobotInfo:
    pos_dir: PosDir
    sensor: Sensor

@dataclass
class DiffDriveValues:
    right_velo: float
    left_velo: float

@dataclass
class Sensor:
    opponent_in_sector: SectorName
    left_distance: float
    front_distance: float
    right_distance: float

class Controller:

    def take_step(self, sensor: Sensor) -> DiffDriveValues:
        pass

def start_duel(controller1: Controller, controller2: Controller) -> None:
    pass