import pytest

import training.value_mapping as vm
from training.simrunner import CombiSensor, SectorName
from training.value_mapping import (
    DiffDriveMapping,
    MappingParameters,
    _cv,
    _diff_drive_values,
)

test_parameters_a = MappingParameters(
    max_speed=40,
    step_count_speed=4,
    max_view_distance=300.0,
    step_count_view_distance=3,
)
test_parameters_b = MappingParameters(
    max_speed=40,
    step_count_speed=4,
    max_view_distance=500.0,
    step_count_view_distance=5,
)

sensor_a_testdata = [
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
        _cv(1, 3) + _cv(1, 3) + _cv(3, 3) + _cv(1, 3),
    ),
    (
        CombiSensor(
            left_distance=290,
            front_distance=350,
            right_distance=400.1,
            opponent_in_sector=SectorName.RIGHT,
        ),
        _cv(2, 3) + _cv(3, 3) + _cv(3, 3) + _cv(2, 3),
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_a_testdata)
def test_sensor_a(sensor: CombiSensor, expected: list[float]):
    result = vm.sensor_to_vector(sensor, test_parameters_a)
    assert result == expected


sensor_b_testdata = [
    (
        CombiSensor(
            left_distance=250.0,
            front_distance=350.0,
            right_distance=501.0,
            opponent_in_sector=SectorName.UNDEF,
        ),
        _cv(2, 5) + _cv(3, 5) + _cv(-1, 5) + _cv(-1, 3),
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_b_testdata)
def test_sensor_b(sensor: CombiSensor, expected: list[float]):
    result = vm.sensor_to_vector(sensor, test_parameters_b)
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


diff_drive_values_testdata = [
    (3.0, 3, [-3.0, 0.0, 3.0]),
    (6.0, 5, [-6.0, -3.0, 0.0, 3.0, 6.0]),
]


@pytest.mark.parametrize("max_value, step_count, expected", diff_drive_values_testdata)
def test_diff_drive_values(max_value: float, step_count: int, expected: list[float]):
    result = _diff_drive_values(max_value, step_count)
    assert result == expected


dd_mapping_a = DiffDriveMapping(2.0, 3)
dd_mapping_b = DiffDriveMapping(7.0, 5)

dd_mapping_a_testdata = [
    (2.00, 3, 0, -2.00, 2.00),
    (2.00, 3, 1, 0.00, 2.00),
    (2.00, 3, 2, -2.00, 0.00),
    (2.00, 3, 3, 0.00, 0.00),
    (2.00, 3, 4, -2.00, -2.00),
    (2.00, 3, 5, 2.00, 2.00),
    (2.00, 3, 6, 0.00, -2.00),
    (2.00, 3, 7, 2.00, 0.00),
    (2.00, 3, 8, 2.00, -2.00),
]


@pytest.mark.parametrize(
    "max_value, step_count, index, left, right", dd_mapping_a_testdata
)
def test_diff_drive_mapping_a(
    max_value: float, step_count: int, index: int, left: float, right: float
):
    result = dd_mapping_a.get(index)
    assert result == (left, right)


dd_mapping_b_testdata = [
    (7.00, 5, 0, -7.00, 7.00),
    (7.00, 5, 1, -3.50, 7.00),
    (7.00, 5, 2, -7.00, 3.50),
    (7.00, 5, 3, 0.00, 7.00),
    (7.00, 5, 4, -3.50, 3.50),
    (7.00, 5, 5, -7.00, 0.00),
    (7.00, 5, 6, 0.00, 3.50),
    (7.00, 5, 7, -3.50, 0.00),
    (7.00, 5, 8, 3.50, 7.00),
    (7.00, 5, 9, -7.00, -3.50),
    (7.00, 5, 10, 0.00, 0.00),
    (7.00, 5, 11, -3.50, -3.50),
    (7.00, 5, 12, 3.50, 3.50),
    (7.00, 5, 13, -7.00, -7.00),
    (7.00, 5, 14, 7.00, 7.00),
    (7.00, 5, 15, 0.00, -3.50),
    (7.00, 5, 16, -3.50, -7.00),
    (7.00, 5, 17, 3.50, 0.00),
    (7.00, 5, 18, 7.00, 3.50),
    (7.00, 5, 19, 0.00, -7.00),
    (7.00, 5, 20, 3.50, -3.50),
    (7.00, 5, 21, 7.00, 0.00),
    (7.00, 5, 22, 3.50, -7.00),
    (7.00, 5, 23, 7.00, -3.50),
    (7.00, 5, 24, 7.00, -7.00),
]


@pytest.mark.parametrize(
    "max_value, step_count, index, left, right", dd_mapping_b_testdata
)
def test_diff_drive_mapping_b(
    max_value: float, step_count: int, index: int, left: float, right: float
):
    result = dd_mapping_b.get(index)
    assert result == (left, right)
