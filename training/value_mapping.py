from dataclasses import dataclass

from training.simrunner import CombiSensor, DiffDriveValues


@dataclass(frozen=True)
class MappingParameters:
    max_speed: float = (7.0,)
    step_count_speed: int = (5,)
    max_view_distance: float = (300.0,)
    step_count_view_distance: int = 4


def sensor_to_vector(sensor: CombiSensor, parameters: MappingParameters) -> list[float]:
    max_dist_rel = 1.0 / parameters.max_view_distance
    k = parameters.step_count_speed / max_dist_rel

    def i_dist(dist: float) -> int:
        dist_rel = dist / parameters.max_view_distance
        return int(k * dist_rel)

    return (
        _cv(i_dist(sensor.left_distance), parameters.step_count_view_distance)
        + _cv(i_dist(sensor.front_distance), parameters.step_count_view_distance)
        + _cv(i_dist(sensor.right_distance), parameters.step_count_view_distance)
        + _cv(-1, 3)
    )


def vector_to_values(
    values: list[float], parameters: MappingParameters
) -> DiffDriveValues:
    return DiffDriveValues(right_velo=0.0, left_velo=0.0)


# Create a vector of 0.0 with 1.0 at index
def _cv(index: int, step_count: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(step_count)]
