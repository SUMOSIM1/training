import subprocess as sp
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import simdb as db
from dataclasses_json import DataClassJsonMixin, dataclass_json


class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


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
class StartCommand(DataClassJsonMixin):
    id: str
    robot1_port: int
    robot2_port: int
    result_port: int


@dataclass
class StartResponse(DataClassJsonMixin):
    ok: bool
    message: Optional[str]


@dataclass_json
@dataclass
class Simulation:
    base_port: int
    started_at: datetime = datetime.now()
    status: str = "running"  # running, finished, timeout, error


def run(simulation_port: int, path: Path):
    if not path.exists():
        raise RuntimeError(f"simpath {path} does not exist.")
    sbtfile = path / "build.sbt"
    if not sbtfile.exists():
        raise RuntimeError(f"simpath {path} contains no 'build.sbt' file.")
    print(f"starting sim in {path}")
    sp.call(
        ["sbt", "--supershell=false", f"sumosimJVM/run udp --port {simulation_port}"],
        cwd=f"{path}",
    )


def start(base_port: int):
    with db.create_client() as client:
        running_sim = db.find_running(client, "running", base_port)
        if running_sim:
            raise RuntimeError(f"Baseport {base_port} is currently running")
        sim = Simulation(
            base_port=base_port,
        )
        id = db.insert(client, sim.to_dict())
        command = StartCommand(
            id=id,
            robot1_port=base_port * 10 + 0,
            robot2_port=base_port * 10 + 1,
            result_port=base_port * 10 + 2,
        )
        print(f"---> starting on {command}")
