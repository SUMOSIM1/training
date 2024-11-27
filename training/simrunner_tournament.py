import itertools as it
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import training.consts
import training.simrunner as sr
import training.sumosim_helper as sh
from training.simrunner import SimInfo


class CombinationType(str, Enum):
    WITH_REPLACEMENT = "with-replacement"
    WITHOUT_REPLACEMENT = "without-replacement"


@dataclass
class Result:
    combinations: list[(str, str)]
    data: list[(str, str, float, float)]


def start(
    port: int,
    name: str,
    controller_names: list[sr.ControllerName],
    reward_handler_name: sr.RewardHandlerName,
    combination_type: CombinationType.WITHOUT_REPLACEMENT,
    max_simulation_steps: int,
    epoch_count: int,
    record: bool,
):
    if max_simulation_steps > training.consts.MAX_STEPS:
        raise ValueError(
            f"The simulator runs maximum {training.consts.MAX_STEPS} steps. "
            f"Defining max-simulation-steps greater than that makes no sense"
        )

    out_dir = Path.home() / "tmp" / "sumosim"
    out_dir.mkdir(exist_ok=True, parents=True)
    for f in out_dir.iterdir():
        if name in f.name:
            raise RuntimeError(f"Name '{name}' was already used")

    combinations = _tournament_combinations(controller_names, combination_type)
    combinations_count = len(combinations)
    combination_nr = 1
    result_data = sh.RewardCollector()
    for controller_name1, controller_name2 in combinations:
        for epoch_nr in range(epoch_count):
            reward1, reward2, msg = run_epoch(
                port,
                name,
                max_simulation_steps,
                combination_nr,
                epoch_nr,
                controller_name1,
                controller_name2,
                reward_handler_name,
                record,
            )
            print(
                f"Finished epoch c:{combination_nr}/{combinations_count} "
                f"e:{epoch_nr + 1}/{epoch_count} "
                f"r1:{reward1:10.2f} r2:{reward2:10.2f} {msg}"
            )
            result_data.add(
                controller_name1.value, controller_name2.value, reward1, reward2
            )
            epoch_nr += 1
        combination_nr += 1

    controller_names_str = ", ".join([c.value for c in controller_names])
    desc = {
        "name": name,
        "controller": controller_names_str,
        "reward handler": reward_handler_name.value,
        "max sim steps": max_simulation_steps,
        "epoch count": epoch_count,
    }
    sh.write_data(result_data, out_dir, name)
    lines = sh.create_lines(desc, [[0, 3, 4], [1], [2]])
    sh.plot_epoch_datas(data=result_data, out_dir=out_dir, name=name, suptitle=lines)
    print(f"Wrote results for {name} to {out_dir}")


def run_epoch(
    port: int,
    name: str,
    max_simulation_steps: int,
    combination_number: int,
    epoch_number: int,
    controller_name1: sr.ControllerName,
    controller_name2: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
    record: bool,
) -> (float, float):
    sim_name = f"{name}-{combination_number:03d}-{epoch_number:04d}"
    controller1 = sr.ControllerProvider.get(controller_name1)
    controller2 = sr.ControllerProvider.get(controller_name2)
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)

    def apply_policies(sensor_response: sr.SensorResponse) -> sr.ActionRequest:
        diff_drive1 = controller1.take_step(sensor_response.sensor1)
        diff_drive2 = controller2.take_step(sensor_response.sensor2)
        return sr.ActionRequest(
            diffDrive1=diff_drive1,
            diffDrive2=diff_drive2,
            cnt=cnt + 1,
            simulation_states=simulation_states,
        )

    sim_info = None
    if record:
        sim_info = SimInfo(
            sim_name=sim_name,
            port=port,
            name1=controller1.name(),
            desc1=controller1.description(),
            name2=controller2.name(),
            desc2=controller2.description(),
            max_simulation_steps=max_simulation_steps,
        )
    try:
        response: sr.Response = sr.reset(port, max_simulation_steps, reward_handler)
        cnt = 0
        cumulative_reward1 = 0.0
        cumulative_reward2 = 0.0
        while True:
            match response:
                case sr.FinishedResponse(reward1, reward2, msg):
                    cumulative_reward1 += reward1
                    cumulative_reward2 += reward2
                    return cumulative_reward1, cumulative_reward2, msg
                case sr.ErrorResponse(message=msg):
                    error_msg = f"ERROR running {sim_name}: {msg}"
                    print(error_msg)
                    raise RuntimeError(error_msg)
                case sr.SensorResponse(
                    reward1=reward1,
                    reward2=reward2,
                    simulation_states=simulation_states,
                    cnt=cnt,
                ):
                    cumulative_reward1 += reward1
                    cumulative_reward2 += reward2
                    # print(f"### {epoch_name} {cnt} reward:{reward}")
                    # noinspection PyTypeChecker
                    request = apply_policies(response)
                    stop = cnt + 1 >= max_simulation_steps
                    response = sr.step(
                        request,
                        reward_handler,
                        port,
                        stop,
                        max_simulation_steps,
                        sim_info,
                    )
                case _:
                    raise RuntimeError(f"Could not match response:{response}")
    except BaseException as ex:
        print(f"### Error running {sim_name} {ex}")
        raise ex


def _tournament_combinations(
    names: list, combination_type: CombinationType
) -> list[(any, any)]:
    match combination_type:
        case CombinationType.WITH_REPLACEMENT:
            return list(it.combinations_with_replacement(names, 2))
        case CombinationType.WITHOUT_REPLACEMENT:
            return list(it.combinations(names, 2))
