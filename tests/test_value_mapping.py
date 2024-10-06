import pytest

import training.value_mapping as vm
from training.simrunner import CombiSensor, SectorName
from training.value_mapping import MappingParameters

test_parameters = MappingParameters(
    max_speed=100,
    step_count_speed=2,
    max_view_distance=300.0,
    step_count_view_distance=4,
)

sensor_testdata = [
    (
        CombiSensor(
            left_distance=0.0,
            front_distance=0.0,
            right_distance=0.0,
            opponent_in_sector=SectorName.UNDEF,
        ),
        [],
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_testdata)
def test_sensor(sensor: CombiSensor, expected: list[float]):
    result = vm.sensor_to_vector(sensor, test_parameters)
    assert result == expected
