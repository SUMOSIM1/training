import math
from dataclasses import dataclass
from typing import Tuple

from training.simrunner import CombiSensor, DiffDriveValues, SectorName


@dataclass(frozen=True)
class MappingParameters:
    max_speed: float = 7.0
    step_count_speed: int = 5
    max_view_distance: float = 300.0
    step_count_view_distance: int = 4


def sensor_to_vector(sensor: CombiSensor, parameters: MappingParameters) -> list[float]:
    k = float(parameters.step_count_view_distance)

    def i_dist(dist: float) -> int:
        dist_rel = dist / parameters.max_view_distance
        if dist < 0.0:
            return -1
        if dist_rel > 1.0:
            return parameters.step_count_view_distance
        print(f"### dist_rel {dist} {dist_rel} | {k} ")
        return int(math.floor(k * dist_rel))

    i_left = i_dist(sensor.left_distance)
    i_front = i_dist(sensor.front_distance)
    i_right = i_dist(sensor.right_distance)
    i_view = -1
    match sensor.opponent_in_sector:
        case SectorName.LEFT:
            i_view = 0
        case SectorName.CENTER:
            i_view = 1
        case SectorName.RIGHT:
            i_view = 2

    print(f"### i dist {i_left} {i_front} {i_right}")
    return (
        _cv(i_left, parameters.step_count_view_distance)
        + _cv(i_front, parameters.step_count_view_distance)
        + _cv(i_right, parameters.step_count_view_distance)
        + _cv(i_view, 3)
    )


def vector_to_values(
    values: list[float], parameters: MappingParameters
) -> DiffDriveValues:
    return DiffDriveValues(right_velo=0.0, left_velo=0.0)


# Create a vector of 0.0 with 1.0 at index
def _cv(index: int, step_count: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(step_count)]


def _diff_drive_values(max_speed: float, max_speed_index: int) -> list[float]:
    assert max_speed_index % 2 == 1
    diff = max_speed / ((max_speed_index - 1) / 2.0)
    return [-max_speed + (i * diff) for i in range(max_speed_index)]


def _diff_drive_tuple(max_speed: float, max_speed_index: int) -> Tuple:
    def sort_key(t) -> float:
        diff = t[0] - t[1]
        norm = abs(t[0]) * 0.001
        return diff + norm

    assert max_speed_index % 2 == 1
    values = _diff_drive_values(max_speed, max_speed_index)
    re = []
    for v1 in values:
        for v2 in values:
            re.append((v1, v2))
    return sorted(re, key=sort_key)


class DiffDriveMapping:
    diff_drive_tuples: Tuple

    def __init__(self, max_speed: float, max_speed_index: int):
        self.diff_drive_tuples = _diff_drive_tuple(max_speed, max_speed_index)

    def get(self, index: int):
        return self.diff_drive_tuples[index]
