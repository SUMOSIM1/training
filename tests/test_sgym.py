from typing import Any

import numpy as np
import pytest

import training.sgym as sgym
import training.sgym_training as sl
import training.simrunner as sr

_config_a = sgym.SEnvConfig(
    max_wheel_speed=100.0, max_view_distance=200.0, max_simulation_steps=1000
)

_as_a = sgym._crete_action_space(_config_a)
_os_a = sgym._create_observation_space(_config_a)

sensor_to_observation_space_a_testdata = [
    (
        sr.CombiSensor(
            left_distance=20,
            front_distance=33,
            right_distance=66,
            opponent_in_sector=sr.SectorName.UNDEF,
        ),
        {
            "view": 0,
            "border": sgym._create_numpy_array([[20, 33, 66]], _config_a),
        },
    ),
    (
        sr.CombiSensor(
            left_distance=0.0,
            front_distance=100.1,
            right_distance=150.0,
            opponent_in_sector=sr.SectorName.UNDEF,
        ),
        {
            "view": 0,
            "border": sgym._create_numpy_array([[0.0, 100.1, 150.0]], _config_a),
        },
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_to_observation_space_a_testdata)
def test_sensor_to_observation_space_a(sensor: sr.CombiSensor, expected: dict):
    result = sgym.mapping_sensor_to_observation_space(sensor, _config_a)
    assert _os_a.contains(result)
    for k in result:
        np.testing.assert_equal(result[k], expected[k], err_msg=f"Comparing {k}")


action_space_to_diff_drive_a_testdata = [
    (sgym._create_numpy_array([[0.1, 0.5]], _config_a), sr.DiffDriveValues(0.1, 0.5)),
    (sgym._create_numpy_array([[-0.1, 0.5]], _config_a), sr.DiffDriveValues(-0.1, 0.5)),
    (sgym._create_numpy_array([[0.1, -0.5]], _config_a), sr.DiffDriveValues(0.1, -0.5)),
    (sgym._create_numpy_array([[35, 45]], _config_a), sr.DiffDriveValues(35, 45)),
    (sgym._create_numpy_array([[-35, 45]], _config_a), sr.DiffDriveValues(-35, 45)),
    (sgym._create_numpy_array([[-35, -45]], _config_a), sr.DiffDriveValues(-35, -45)),
]


@pytest.mark.parametrize("space, expected", action_space_to_diff_drive_a_testdata)
def test_action_space_to_diff_drive_a(space: Any, expected: sr.DiffDriveValues):
    result = sgym.mapping_action_space_to_diff_drive(space)
    assert _as_a.contains(space)
    assert result == expected


continuous_to_discrete_testdata = [
    (-10.0, 10.0, 3, (-100.0001, 0)),
    (-10.0, 10.0, 3, (-10.0001, 0)),
    (-10.0, 10.0, 3, (-10.0, 0)),
    (-10.0, 10.0, 3, (-3.334, 0)),
    (-10.0, 10.0, 3, (-3.332, 1)),
    (-10.0, 10.0, 3, (0.0, 1)),
    (-10.0, 10.0, 3, (3.332, 1)),
    (-10.0, 10.0, 3, (3.334, 2)),
    (-10.0, 10.0, 3, (4, 2)),
    (-10.0, 10.0, 3, (10.0, 2)),
    (-10.0, 10.0, 3, (10.0001, 2)),
    (-10.0, 10.0, 3, (100.0001, 2)),
]


@pytest.mark.parametrize(
    "min_value, max_value, n, expected", continuous_to_discrete_testdata
)
def test_continuous_to_discrete(
    min_value: float, max_value: float, n: int, expected: tuple
):
    v, i = expected
    result = sl._continuous_to_discrete(v, min_value, max_value, n)
    print(f"### i:{i} r:{result}")
    assert i == result


create_subset_testdata = [
    (2.0, 2, [-2.0, -1.0, 0.0, 1.0, 2.0]),
    (2.0, 4, [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
]


@pytest.mark.parametrize("max_value,n, expected", create_subset_testdata)
def test_x(max_value: float, n: int, expected: list[float]):
    result = sl._create_subset(max_value, n)
    assert result == expected
