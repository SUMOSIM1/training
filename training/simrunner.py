from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json

import training.simdb as db
import training.udp as udp
import training.util as util

import training.controller as ctl


class SendCommand:
    pass


class ReceiveCommand:
    pass


@dataclass_json
@dataclass
class PosDir:
    xpos: float
    ypos: float
    direction: float


@dataclass
class SensorDto:
    pos_dir: PosDir
    combi_sensor: ctl.CombiSensor


@dataclass
class StartCommand(SendCommand):
    pass


@dataclass
class SensorCommand(ReceiveCommand):
    robot1_sensor: SensorDto
    robot2_sensor: SensorDto


@dataclass
class DiffDriveCommand(SendCommand):
    robot1_diff_drive_values: ctl.DiffDriveValues
    robot2_diff_drive_values: ctl.DiffDriveValues


@dataclass
class FinishedOkCommand(ReceiveCommand):
    robot1_rewards: list[(str, str)]
    robot2_rewards: list[(str, str)]


@dataclass_json
@dataclass
class SimulationState:
    robot1: PosDir
    robot2: PosDir


@dataclass
class FinishedErrorCommand(ReceiveCommand):
    message: str


def start(port: int):
    controller1 = ctl.ControllerProvider.get("slow-circle")
    controller2 = ctl.ControllerProvider.get("fast-circle")

    simulation_states = []

    with db.create_client() as client:

        def check_running():
            running_sim = db.find_running(client, "running", port)
            print(f"--- Found running for {port} {running_sim}")
            if running_sim:
                raise RuntimeError(f"Baseport {port} is currently running")

        def insert_new_sim() -> str:
            sim_name = f"{controller1.name()} : {controller2.name()}"
            sim_description = {
                "controller1": controller1.description(),
                "controller2": controller2.description(),
            }
            sim = db.Simulation(
                port=port,
                name=sim_name,
                description=sim_description,
            )
            _obj_id = db.insert(client, sim.to_dict())
            print(f"--- Wrote to database id:{_obj_id} sim:{sim}")
            return _obj_id

        def send_command_and_wait(command: SendCommand) -> ReceiveCommand:
            send_str = format_command(command)
            print(f"---> Sending {command} - {send_str}")
            resp_str = udp.send_and_wait(send_str, port, 10)
            resp = parse_command(resp_str)
            print(f"<--- Result {resp} {resp_str}")
            return resp

        check_running()
        obj_id = insert_new_sim()
        try:
            command = StartCommand()
            cnt = 0
            while True:
                cnt += 1
                print("")
                response: ReceiveCommand = send_command_and_wait(command)
                match response:
                    case SensorCommand(s1, s2):
                        print("sensors", s1, s2)
                        state = SimulationState(s1.pos_dir, s2.pos_dir)
                        simulation_states.append(state)

                        r1 = controller1.take_step(s1)
                        r2 = controller1.take_step(s2)

                        command = DiffDriveCommand(r1, r2)
                    case FinishedOkCommand(r1, r2):
                        events_dict = {"r1": r1, "r2": r2}
                        state_list = [state.to_json() for state in simulation_states]
                        db.update_status_finished(
                            client, obj_id, events_dict, state_list
                        )
                        print(
                            f"Finished with OK: {obj_id} {events_dict}"
                            f"{state_list[:5]}..."
                        )
                        break
                    case FinishedErrorCommand(msg):
                        db.update_status_error(client, obj_id, msg)
                        print(f"Finished with ERROR: {obj_id} {msg}")
                        break

        except BaseException as ex:
            msg = util.message(ex)
            print(f"ERROR: {msg}")
            db.update_status_error(client, obj_id, msg)


def format_command(cmd: SendCommand) -> str:
    def format_float(value: float) -> str:
        return f"{value:.4f}"

    def format_diff_drive_values(values: ctl.DiffDriveValues) -> str:
        return f"{format_float(values.left_velo)};{format_float(values.right_velo)}"

    match cmd:
        case StartCommand():
            return "A|"
        case DiffDriveCommand(r1, r2):
            return f"C|{format_diff_drive_values(r1)}#{format_diff_drive_values(r2)}"
        case _:
            raise NotImplementedError(f"format_command {cmd}")


def parse_command(data: str) -> ReceiveCommand:
    def parse_sensor_dto(sensor_data: str) -> SensorDto:
        ds = sensor_data.split(";")
        return SensorDto(
            pos_dir=PosDir(float(ds[0]), float(ds[1]), float(ds[2])),
            combi_sensor=ctl.CombiSensor(
                left_distance=float(ds[3]),
                front_distance=float(ds[4]),
                right_distance=float(ds[5]),
                opponent_in_sector=ctl.SectorName[ds[6]],
            ),
        )

    def parse_finished(data: str) -> SensorDto:
        print(f"--- data '{data}'")
        if data:
            ds = data.split(";")
            return [(d.split("!")[0], d.split("!")[1]) for d in ds]
        else:
            return []

    (h, d) = data.split("|")
    match h:
        case "E":
            return FinishedErrorCommand(d)
        case "B":
            (r1, r2) = d.split("#")
            return SensorCommand(parse_sensor_dto(r1), parse_sensor_dto(r2))
        case "D":
            (r1, r2) = d.split("#")
            return FinishedOkCommand(parse_finished(r1), parse_finished(r2))
        case _:
            raise NotImplementedError(f"parse_command {data}")
