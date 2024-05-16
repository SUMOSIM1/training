from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import simdb as db
import udp
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


def start(simulation_port, base_port: int):
    with db.create_client() as client:
        running_sim = db.find_running(client, "running", base_port)
        print(f"--- Found running for {base_port} {running_sim}")
        if running_sim:
            raise RuntimeError(f"Baseport {base_port} is currently running")
        sim = Simulation(
            base_port=base_port,
        )
        id = db.insert(client, sim.to_dict())
        print(f"--- Wrote to databas id:{id} sim:{sim}")
        try:
            command = StartCommand(
                id=id,
                robot1port=base_port * 10 + 0,
                robot2port=base_port * 10 + 1,
            )
            print(f"---> Sending {command}")
            answer_str = udp.send_and_wait(command.to_json(), simulation_port, 5)
            answer = dataclass_json(StartResponse).from_json(answer_str)
            print(f"<--- Result {answer}")
        except BaseException as ex:
            msg = util.message(ex)
            print(f"ERROR: {msg}")
            db.update_status(client, id, "error", msg)
