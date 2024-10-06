from dataclasses import dataclass

from training.simrunner import CombiSensor, DiffDriveValues


@dataclass(frozen=True)
class MappingParameters:
    max_speed: float = (7.0,)
    step_count_speed: int = (5,)
    max_view_distance: float = (300.0,)
    step_count_view_distance: int = 4


def sensor_to_vector(sensor: CombiSensor, parameters: MappingParameters) -> list[float]:
    return []


def vector_to_values(
    values: list[float], parameters: MappingParameters
) -> DiffDriveValues:
    return DiffDriveValues(right_velo=0.0, left_velo=0.0)
