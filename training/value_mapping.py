import math
from dataclasses import dataclass

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
