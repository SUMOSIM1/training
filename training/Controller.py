from enum import Enum, auto
from dataclasses_json import dataclass_json, DataClassJsonMixin
from dataclasses import dataclass
import json
import UdpClient as udp

class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class Command(Enum):
    START = "START"
    STOP = "STOP"

@dataclass_json
@dataclass
class Point2:
    xpos: float
    ypos: float

@dataclass_json
@dataclass
class PosDir:
    pos: Point2
    dir: float

@dataclass
class Sensor(DataClassJsonMixin):
    opponent_in_sector: SectorName
    left_distance: float
    front_distance: float
    right_distance: float

@dataclass_json
@dataclass
class RobotInfo:
    pos_dir: PosDir
    sensor: Sensor

@dataclass_json
@dataclass
class DiffDriveValues:
    right_velo: float
    left_velo: float

@dataclass
class CommandDto(DataClassJsonMixin):
    command: Command

def sendAndWait(cmd: Command, config: udp.Config) -> any:
    j = cmd.to_json()
    udp.sendAndWait(j, config)



cmd = CommandDto(Command.START)
config = udp.Config()
sendAndWait(cmd, config)
