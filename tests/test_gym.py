from gymnasium import Space

import training.gym as sgym
from training.simrunner import CombiSensor, SectorName, DiffDriveValues


def test_sensor_to_observation_space_a():
    sensor = CombiSensor(
        left_distance=200.0,
        front_distance=100.0,
        right_distance=150.0,
        opponent_in_sector=SectorName.UNDEF
    )
    space = sgym.sensor_to_observation_space(sensor)
    print("## space", space)


def test_action_space_to_w_a(action: Space)-> DiffDriveValues:
    action = None
    diff_drive = sgym.action_space_to_diff_drive(action)
    print("## diff_drive", diff_drive)