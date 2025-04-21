from collections.abc import Callable

import gymnasium as gym
import gymnasium.spaces as gyms

import training.helper as hlp
import training.sgym.core as sgym
import training.simrunner as sr

from enum import Enum


class SEnvMappingName(str, Enum):
    LINEAR = ("linear",)
    NON_LINEAR_1 = ("non-linear-1",)
    NON_LINEAR_2 = ("non-linear-2",)
    NON_LINEAR_3 = ("non-linear-3",)
    NON_LINEAR_4 = ("non-linear-4",)


def senv_mapping(mapping: SEnvMappingName, cfg: sgym.SEnvConfig) -> sgym.SEnvMapping:
    def create_senv_mapping(linear: int) -> sgym.SEnvMapping:
        return sgym.SEnvMapping(
            act_space=_q_act_space,
            obs_space=_q_obs_space,
            map_act=_fun_map_q_act_to_diff_drive(cfg),
            map_sensor=_fun_map_q_sensor_to_obs(linear),
        )

    match mapping:
        case SEnvMappingName.LINEAR:
            return create_senv_mapping(0)
        case SEnvMappingName.NON_LINEAR_1:
            return create_senv_mapping(-1)
        case SEnvMappingName.NON_LINEAR_2:
            return create_senv_mapping(-2)
        case SEnvMappingName.NON_LINEAR_3:
            return create_senv_mapping(-3)
        case SEnvMappingName.NON_LINEAR_4:
            return create_senv_mapping(-4)
        case _:
            raise ValueError(f"Unknown SEnvMappingName {mapping}")


def _q_act_space(config: sgym.SEnvConfig) -> gym.Space:
    n = (config.wheel_speed_steps + 1) * (config.wheel_speed_steps + 1)
    return gyms.Discrete(n)


def _q_obs_space(config: sgym.SEnvConfig) -> gym.Space:
    return gyms.Tuple(
        (
            gyms.Discrete(n=4),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
        )
    )


def _fun_map_q_act_to_diff_drive(
    config: sgym.SEnvConfig,
) -> Callable[[any, sgym.SEnvConfig], sr.DiffDriveValues]:
    velo_from_index = _fun_velo_from_index(
        config.max_wheel_speed, config.wheel_speed_steps
    )

    def inner(a_space: int, _config: sgym.SEnvConfig) -> sr.DiffDriveValues:
        return velo_from_index(a_space)

    return inner


def _fun_map_q_sensor_to_obs(
    linear: int,
) -> Callable[[sr.CombiSensor, sgym.SEnvConfig], tuple[int, int, int, int]]:
    def inner(
        sensor: sr.CombiSensor, config: sgym.SEnvConfig
    ) -> tuple[int, int, int, int]:
        def discrete(distance: float) -> int:
            return hlp.cont_to_discrete(
                distance,
                0.0,
                config.max_view_distance,
                config.view_distance_steps,
                linear,
            )

        return (
            sr.sector_mapping(sensor.opponent_in_sector),
            discrete(sensor.left_distance),
            discrete(sensor.front_distance),
            discrete(sensor.right_distance),
        )

    return inner


def _fun_velo_from_index(
    max_velo: float, velo_steps: int
) -> Callable[[int], sr.DiffDriveValues]:
    velos = hlp.cont_values(-max_velo, max_velo, velo_steps + 1)
    n = len(velos)
    diff_drives = []
    for i in range(n):
        for j in range(n):
            diff_drives.append(sr.DiffDriveValues(velos[i], velos[j]))

    def inner(index: int) -> sr.DiffDriveValues:
        return diff_drives[index]

    return inner
