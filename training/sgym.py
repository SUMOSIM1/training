from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

import training.simrunner as sr


@dataclass(frozen=True)
class SEnvConfig:
    max_wheel_speed: float
    max_view_distance: float
    dtype: np.generic = np.float32


class SEnv(gym.Env):
    pass


# Define action and observation space
def crete_action_space(config: SEnvConfig) -> gym.Space:
    return Box(
        low=-config.max_wheel_speed,
        high=config.max_wheel_speed,
        shape=(1, 2),
        dtype=config.dtype,
    )


def create_observation_space(config: SEnvConfig) -> gym.Space:
    observation_view_space = Discrete(n=4)
    observation_border_space = Box(
        low=0.0, high=config.max_view_distance, shape=(1, 3), dtype=config.dtype
    )
    return Dict(
        {
            "view": observation_view_space,
            "border": observation_border_space,
        }
    )


def mapping_sensor_to_observation_space(
    sensor: sr.CombiSensor, config: SEnvConfig
) -> dict[str, any]:
    def view_mapping() -> int:
        match sensor.opponent_in_sector:
            case sr.SectorName.UNDEF:
                return 0
            case sr.SectorName.LEFT:
                return 1
            case sr.SectorName.CENTER:
                return 2
            case sr.SectorName.RIGHT:
                return 3
            case _:
                raise ValueError(f"Wrong sector name {sensor.opponent_in_sector}")

    return {
        "view": view_mapping(),
        "border": _cna(
            [
                [
                    sensor.left_distance,
                    sensor.front_distance,
                    sensor.right_distance,
                ]
            ],
            config,
        ),
    }


def mapping_action_space_to_diff_drive(action_space: list[list]) -> sr.DiffDriveValues:
    return sr.DiffDriveValues(
        left_velo=action_space[0][1],
        right_velo=action_space[0][0],
    )


def _cna(value: Any, config: SEnvConfig) -> np.array:
    return np.array(value, dtype=config.dtype)
