from dataclasses import dataclass
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
    id: str
    messages: list[str]


def start(simulation_port, base_port: int):
    with db.create_client() as client:

        def check_running():
            running_sim = db.find_running(client, "running", base_port)
            print(f"--- Found running for {base_port} {running_sim}")
            if running_sim:
                raise RuntimeError(f"Baseport {base_port} is currently running")

        def insert_new_sim() -> str:
            sim = db.Simulation(
                base_port=base_port,
            )
            id = db.insert(client, sim.to_dict())
            print(f"--- Wrote to databas id:{id} sim:{sim}")
            return id

        def start_sim_and_wait(
            id: str, timeout_sec: int, robot1port: int, robot2port: int
        ) -> str:
            command = StartCommand(
                id=id,
                robot1port=robot1port,
                robot2port=robot2port,
            )
            print(f"---> Sending {command}")
            resp_str = udp.send_and_wait(
                command.to_json(), simulation_port, timeout_sec
            )
            resp = dataclass_json(StartResponse).from_json(resp_str)
            print(f"<--- Result {resp}")
            if not resp.ok:
                msg = ", ".join(resp.messages)
                db.update_status(client, resp.id, db.SIM_STATUS_ERROR, msg)
                return f"Not OK: {msg}"
            else:
                db.update_status(client, resp.id, db.SIM_STATUS_FINISHED, "")
                return ""

        def robot1handler(data: str) -> str:
            return "# robo 1 #"

        def robot2handler(data: str) -> str:
            return "# robo 2 #"

        check_running()
        id = insert_new_sim()
        try:
            robot1port = base_port * 10 + 0
            robot2port = base_port * 10 + 1
            # start concurrent here
            functions = [
                (start_sim_and_wait, id, 4, robot1port, robot2port),
                (udp.open_socket, robot1port, 5, robot1handler),
                (udp.open_socket, robot2port, 5, robot2handler),
            ]
            results = util.run_concurrent(functions, 10)
            if results:
                for result in results:
                    print(f"## Error calling concurrent [{result}]")
        except BaseException as ex:
            msg = util.message(ex)
            print(f"ERROR: {msg}")
            db.update_status(client, id, db.SIM_STATUS_ERROR, msg)
