import subprocess as sp
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import simdb as db
import UdpClient
import util
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
    robot1port: int
    robot2port: int
    resultport: int


@dataclass
class StartResponse(DataClassJsonMixin):
    ok: bool
    messages: list[str]


@dataclass_json
@dataclass
class Simulation:
    base_port: int
    started_at: datetime = datetime.now()
    status: str = "running"  # running, finished, timeout, error
    message: str = ""


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
    def send(client: ABC, obj_id: str):
        try:
            command = StartCommand(
                id=id,
                robot1port=base_port * 10 + 0,
                robot2port=base_port * 10 + 1,
                resultport=base_port * 10 + 2,
            )
            print(f"---> sending {command}")
            answer = UdpClient.send_and_wait(command.to_json(), 4444)
            print(f"<--- {answer}")
            answer1 = dataclass_json(StartResponse).from_json(answer)
            print(f"<--- {answer1}")
            if not answer1.ok:
                db.update_status(client, obj_id, "error", ", ".join(answer1.messages))
            # TODO continue here
        except BaseException as ex:
            msg = util.message(ex)
            print(f"ERROR: {msg}")
            db.update_status(client, obj_id, "error", msg)

    with db.create_client() as client:
        running_sim = db.find_running(client, "running", base_port)
        if running_sim:
            raise RuntimeError(f"Baseport {base_port} is currently running")
        sim = Simulation(
            base_port=base_port,
        )
        id = db.insert(client, sim.to_dict())
        send(client, id)
