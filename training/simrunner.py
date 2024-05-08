import subprocess as sp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import UdpClient as udp
from dataclasses_json import DataClassJsonMixin, dataclass_json


class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Command(Enum):
    START = "START"
    STOP = "STOP"


@dataclass
class SensorDto(DataClassJsonMixin):
    xpos: float
    ypos: float
    direction: float
    opponentinsector: SectorName
    leftdistance: float
    frontdistance: float
    rightdistance: float


@dataclass_json
@dataclass
class DiffDriveValuesDto:
    rightvelo: float
    leftvelo: float


@dataclass
class CommandDto(DataClassJsonMixin):
    command: Command


def sendAndWait(cmd: Command, config: udp.Config) -> any:
    j = cmd.to_json()
    udp.sendAndWait(j, config)


cmd = CommandDto(Command.START)
config = udp.Config()
sendAndWait(cmd, config)


def run(path: Path):
    if not path.exists():
        raise RuntimeError(f"simpath {path} does not exist.")
    sbtfile = path / "build.sbt"
    if not sbtfile.exists():
        raise RuntimeError(f"simpath {path} contains no 'build.sbt' file.")
    print(f"starting sim in {path}")
    sp.call(
        ["sbt", "--supershell=false", "sumosimJVM/run udp --port 4000"], cwd=f"{path}"
    )
