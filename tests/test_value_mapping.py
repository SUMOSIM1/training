import pytest

import training.value_mapping as vm
from training.simrunner import CombiSensor, SectorName
from training.value_mapping import MappingParameters, _cv

test_parameters_a = MappingParameters(
    max_speed=40,
    step_count_speed=4,
    max_view_distance=300.0,
    step_count_view_distance=3,
)


sensor_testdata = [
    (
        CombiSensor(
            left_distance=0.0,
            front_distance=0.0,
            right_distance=0.0,
            opponent_in_sector=SectorName.UNDEF,
        ),
        _cv(0, 3) + _cv(0, 3) + _cv(0, 3) + _cv(-1, 3),
    ),
    (
        CombiSensor(
            left_distance=1.0,
            front_distance=100.1,
            right_distance=150,
            opponent_in_sector=SectorName.UNDEF,
        ),
        _cv(0, 3) + _cv(1, 3) + _cv(1, 3) + _cv(-1, 3),
    ),
    (
        CombiSensor(
            left_distance=99.9,
            front_distance=199.9,
            right_distance=200.1,
            opponent_in_sector=SectorName.LEFT,
        ),
        _cv(0, 3) + _cv(1, 3) + _cv(2, 3) + _cv(0, 3),
    ),
    (
        CombiSensor(
            left_distance=160,
            front_distance=150,
            right_distance=399.9,
            opponent_in_sector=SectorName.CENTER,
        ),
        _cv(0, 3) + _cv(1, 3) + _cv(2, 3) + _cv(1, 3),
    ),
    (
        CombiSensor(
            left_distance=290,
            front_distance=350,
            right_distance=400.1,
            opponent_in_sector=SectorName.RIGHT,
        ),
        _cv(1, 3) + _cv(2, 3) + _cv(-1, 3) + _cv(2, 3),
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_testdata)
def test_sensor(sensor: CombiSensor, expected: list[float]):
    result = vm.sensor_to_vector(sensor, test_parameters_a)
    assert result == expected


cv_testdata = [
    (-100, 4, [0.0, 0.0, 0.0, 0.0]),
    (-1, 4, [0.0, 0.0, 0.0, 0.0]),
    (0, 4, [1.0, 0.0, 0.0, 0.0]),
    (1, 4, [0.0, 1.0, 0.0, 0.0]),
    (2, 4, [0.0, 0.0, 1.0, 0.0]),
    (3, 4, [0.0, 0.0, 0.0, 1.0]),
    (4, 4, [0.0, 0.0, 0.0, 0.0]),
    (100, 4, [0.0, 0.0, 0.0, 0.0]),
    (-1, 2, [0.0, 0.0]),
    (0, 2, [1.0, 0.0]),
    (1, 2, [0.0, 1.0]),
    (2, 2, [0.0, 0.0]),
]


@pytest.mark.parametrize("index, step_count, expected", cv_testdata)
def test_cv(index: int, step_count: int, expected: list[float]):
    result = _cv(index, step_count)
    assert result == expected
