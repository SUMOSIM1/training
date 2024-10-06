from pathlib import Path
from typing import Tuple

import pytest

import training.reward as rw
import training.reward_helper as rwh
import training.reward_util as rwu
from training.reward import RobotEventsEnd, RobotEventsResult

can_see_testdata = [
    ("can_see_T3_A1_0", (0.00, 0.00, 0.00), (250.00, 75.00, 4.70), None),
    ("can_see_T3_A1_1", (0.00, 0.00, 0.00), (250.00, 50.00, 6.27), None),
    ("can_see_T3_A1_2", (0.00, 0.00, 0.00), (250.00, 25.00, 5.94), 251.25),
    ("can_see_T3_A1_3", (0.00, 0.00, 0.00), (250.00, 0.00, 0.33), 250),
    ("can_see_T3_A1_4", (0.00, 0.00, 0.00), (250.00, -25.00, 1.60), 251.25),
    ("can_see_T3_A1_5", (0.00, 0.00, 0.00), (250.00, -50.00, 3.50), None),
    ("can_see_T3_A1_6", (0.00, 0.00, 0.00), (250.00, -75.00, 1.95), None),
    ("can_see_T3_A2_0", (0.00, 0.00, 0.00), (-125.00, 375.00, 3.48), None),
    ("can_see_T3_A2_1", (0.00, 0.00, 0.00), (-100.00, 375.00, 3.15), None),
    ("can_see_T3_A2_2", (0.00, 0.00, 0.00), (-75.00, 375.00, 5.88), None),
    ("can_see_T3_A2_3", (0.00, 0.00, 0.00), (-50.00, 375.00, 1.39), None),
    ("can_see_T3_A2_4", (0.00, 0.00, 0.00), (-25.00, 375.00, 0.17), None),
    ("can_see_T3_A2_5", (0.00, 0.00, 0.00), (0.00, 375.00, 5.77), None),
    ("can_see_T3_A2_6", (0.00, 0.00, 0.00), (25.00, 375.00, 4.82), None),
    ("can_see_T3_A2_7", (0.00, 0.00, 0.00), (50.00, 375.00, 1.87), None),
    ("can_see_T3_A2_8", (0.00, 0.00, 0.00), (75.00, 375.00, 6.10), None),
    ("can_see_T3_A3_0", (0.00, 0.00, 0.00), (-225.00, 75.00, 5.28), None),
    ("can_see_T3_A3_1", (0.00, 0.00, 0.00), (-225.00, 50.00, 5.40), None),
    ("can_see_T3_A3_2", (0.00, 0.00, 0.00), (-225.00, 25.00, 3.22), None),
    ("can_see_T3_A3_3", (0.00, 0.00, 0.00), (-225.00, 0.00, 1.80), None),
    ("can_see_T3_A3_4", (0.00, 0.00, 0.00), (-225.00, -25.00, 6.09), None),
    ("can_see_T3_A3_5", (0.00, 0.00, 0.00), (-225.00, -50.00, 0.54), None),
    ("can_see_T3_A3_6", (0.00, 0.00, 0.00), (-225.00, -75.00, 4.80), None),
    ("can_see_T3_A4_0", (0.00, 0.00, 0.00), (-50.00, -150.00, 1.67), None),
    ("can_see_T3_A4_1", (0.00, 0.00, 0.00), (-25.00, -150.00, 5.45), None),
    ("can_see_T3_A4_2", (0.00, 0.00, 0.00), (0.00, -150.00, 1.09), None),
    ("can_see_T3_A4_3", (0.00, 0.00, 0.00), (25.00, -150.00, 2.73), None),
    ("can_see_T3_A4_4", (0.00, 0.00, 0.00), (50.00, -150.00, 1.34), None),
    ("can_see_T3_B1_0", (0.00, 0.00, 1.57), (250.00, 75.00, 3.20), None),
    ("can_see_T3_B1_1", (0.00, 0.00, 1.57), (250.00, 50.00, 5.02), None),
    ("can_see_T3_B1_2", (0.00, 0.00, 1.57), (250.00, 25.00, 5.02), None),
    ("can_see_T3_B1_3", (0.00, 0.00, 1.57), (250.00, 0.00, 0.12), None),
    ("can_see_T3_B1_4", (0.00, 0.00, 1.57), (250.00, -25.00, 6.25), None),
    ("can_see_T3_B1_5", (0.00, 0.00, 1.57), (250.00, -50.00, 4.42), None),
    ("can_see_T3_B1_6", (0.00, 0.00, 1.57), (250.00, -75.00, 6.17), None),
    ("can_see_T3_B2_0", (0.00, 0.00, 1.57), (-125.00, 375.00, 4.78), None),
    ("can_see_T3_B2_1", (0.00, 0.00, 1.57), (-100.00, 375.00, 2.87), None),
    ("can_see_T3_B2_2", (0.00, 0.00, 1.57), (-75.00, 375.00, 0.92), None),
    ("can_see_T3_B2_3", (0.00, 0.00, 1.57), (-50.00, 375.00, 3.69), 378.32),
    ("can_see_T3_B2_4", (0.00, 0.00, 1.57), (-25.00, 375.00, 3.72), 375.83),
    ("can_see_T3_B2_5", (0.00, 0.00, 1.57), (0.00, 375.00, 1.16), 375.0),
    ("can_see_T3_B2_6", (0.00, 0.00, 1.57), (25.00, 375.00, 2.30), 375.83),
    ("can_see_T3_B2_7", (0.00, 0.00, 1.57), (50.00, 375.00, 3.10), 378.32),
    ("can_see_T3_B2_8", (0.00, 0.00, 1.57), (75.00, 375.00, 0.34), None),
    ("can_see_T3_B3_0", (0.00, 0.00, 1.57), (-225.00, 75.00, 5.75), None),
    ("can_see_T3_B3_1", (0.00, 0.00, 1.57), (-225.00, 50.00, 4.54), None),
    ("can_see_T3_B3_2", (0.00, 0.00, 1.57), (-225.00, 25.00, 2.79), None),
    ("can_see_T3_B3_3", (0.00, 0.00, 1.57), (-225.00, 0.00, 5.89), None),
    ("can_see_T3_B3_4", (0.00, 0.00, 1.57), (-225.00, -25.00, 3.96), None),
    ("can_see_T3_B3_5", (0.00, 0.00, 1.57), (-225.00, -50.00, 0.14), None),
    ("can_see_T3_B3_6", (0.00, 0.00, 1.57), (-225.00, -75.00, 5.19), None),
    ("can_see_T3_B4_0", (0.00, 0.00, 1.57), (-50.00, -150.00, 3.63), None),
    ("can_see_T3_B4_1", (0.00, 0.00, 1.57), (-25.00, -150.00, 3.54), None),
    ("can_see_T3_B4_2", (0.00, 0.00, 1.57), (0.00, -150.00, 2.92), None),
    ("can_see_T3_B4_3", (0.00, 0.00, 1.57), (25.00, -150.00, 4.11), None),
    ("can_see_T3_B4_4", (0.00, 0.00, 1.57), (50.00, -150.00, 0.26), None),
    ("can_see_T3_C1_0", (0.00, 0.00, 3.14), (250.00, 75.00, 4.84), None),
    ("can_see_T3_C1_1", (0.00, 0.00, 3.14), (250.00, 50.00, 0.99), None),
    ("can_see_T3_C1_2", (0.00, 0.00, 3.14), (250.00, 25.00, 1.02), None),
    ("can_see_T3_C1_3", (0.00, 0.00, 3.14), (250.00, 0.00, 5.55), None),
    ("can_see_T3_C1_4", (0.00, 0.00, 3.14), (250.00, -25.00, 5.37), None),
    ("can_see_T3_C1_5", (0.00, 0.00, 3.14), (250.00, -50.00, 1.42), None),
    ("can_see_T3_C1_6", (0.00, 0.00, 3.14), (250.00, -75.00, 1.15), None),
    ("can_see_T3_C2_0", (0.00, 0.00, 3.14), (-125.00, 375.00, 5.19), None),
    ("can_see_T3_C2_1", (0.00, 0.00, 3.14), (-100.00, 375.00, 3.73), None),
    ("can_see_T3_C2_2", (0.00, 0.00, 3.14), (-75.00, 375.00, 4.23), None),
    ("can_see_T3_C2_3", (0.00, 0.00, 3.14), (-50.00, 375.00, 2.71), None),
    ("can_see_T3_C2_4", (0.00, 0.00, 3.14), (-25.00, 375.00, 4.26), None),
    ("can_see_T3_C2_5", (0.00, 0.00, 3.14), (0.00, 375.00, 3.89), None),
    ("can_see_T3_C2_6", (0.00, 0.00, 3.14), (25.00, 375.00, 5.15), None),
    ("can_see_T3_C2_7", (0.00, 0.00, 3.14), (50.00, 375.00, 5.83), None),
    ("can_see_T3_C2_8", (0.00, 0.00, 3.14), (75.00, 375.00, 0.05), None),
    ("can_see_T3_C3_0", (0.00, 0.00, 3.14), (-225.00, 75.00, 4.40), None),
    ("can_see_T3_C3_1", (0.00, 0.00, 3.14), (-225.00, 50.00, 6.27), None),
    ("can_see_T3_C3_2", (0.00, 0.00, 3.14), (-225.00, 25.00, 1.14), 226.38),
    ("can_see_T3_C3_3", (0.00, 0.00, 3.14), (-225.00, 0.00, 3.25), 225),
    ("can_see_T3_C3_4", (0.00, 0.00, 3.14), (-225.00, -25.00, 0.46), 226.38),
    ("can_see_T3_C3_5", (0.00, 0.00, 3.14), (-225.00, -50.00, 4.40), None),
    ("can_see_T3_C3_6", (0.00, 0.00, 3.14), (-225.00, -75.00, 5.25), None),
    ("can_see_T3_C4_0", (0.00, 0.00, 3.14), (-50.00, -150.00, 3.03), None),
    ("can_see_T3_C4_1", (0.00, 0.00, 3.14), (-25.00, -150.00, 5.90), None),
    ("can_see_T3_C4_2", (0.00, 0.00, 3.14), (0.00, -150.00, 4.83), None),
    ("can_see_T3_C4_3", (0.00, 0.00, 3.14), (25.00, -150.00, 1.76), None),
    ("can_see_T3_C4_4", (0.00, 0.00, 3.14), (50.00, -150.00, 3.24), None),
    ("can_see_T3_D1_0", (0.00, 0.00, -1.57), (250.00, 75.00, 5.95), None),
    ("can_see_T3_D1_1", (0.00, 0.00, -1.57), (250.00, 50.00, 4.99), None),
    ("can_see_T3_D1_2", (0.00, 0.00, -1.57), (250.00, 25.00, 0.71), None),
    ("can_see_T3_D1_3", (0.00, 0.00, -1.57), (250.00, 0.00, 4.85), None),
    ("can_see_T3_D1_4", (0.00, 0.00, -1.57), (250.00, -25.00, 1.47), None),
    ("can_see_T3_D1_5", (0.00, 0.00, -1.57), (250.00, -50.00, 0.16), None),
    ("can_see_T3_D1_6", (0.00, 0.00, -1.57), (250.00, -75.00, 2.27), None),
    ("can_see_T3_D2_0", (0.00, 0.00, -1.57), (-125.00, 375.00, 5.22), None),
    ("can_see_T3_D2_1", (0.00, 0.00, -1.57), (-100.00, 375.00, 2.98), None),
    ("can_see_T3_D2_2", (0.00, 0.00, -1.57), (-75.00, 375.00, 2.96), None),
    ("can_see_T3_D2_3", (0.00, 0.00, -1.57), (-50.00, 375.00, 6.13), None),
    ("can_see_T3_D2_4", (0.00, 0.00, -1.57), (-25.00, 375.00, 3.42), None),
    ("can_see_T3_D2_5", (0.00, 0.00, -1.57), (0.00, 375.00, 1.14), None),
    ("can_see_T3_D2_6", (0.00, 0.00, -1.57), (25.00, 375.00, 1.12), None),
    ("can_see_T3_D2_7", (0.00, 0.00, -1.57), (50.00, 375.00, 4.26), None),
    ("can_see_T3_D2_8", (0.00, 0.00, -1.57), (75.00, 375.00, 6.28), None),
    ("can_see_T3_D3_0", (0.00, 0.00, -1.57), (-225.00, 75.00, 4.87), None),
    ("can_see_T3_D3_1", (0.00, 0.00, -1.57), (-225.00, 50.00, 5.35), None),
    ("can_see_T3_D3_2", (0.00, 0.00, -1.57), (-225.00, 25.00, 3.74), None),
    ("can_see_T3_D3_3", (0.00, 0.00, -1.57), (-225.00, 0.00, 2.04), None),
    ("can_see_T3_D3_4", (0.00, 0.00, -1.57), (-225.00, -25.00, 3.00), None),
    ("can_see_T3_D3_5", (0.00, 0.00, -1.57), (-225.00, -50.00, 0.11), None),
    ("can_see_T3_D3_6", (0.00, 0.00, -1.57), (-225.00, -75.00, 6.01), None),
    ("can_see_T3_D4_0", (0.00, 0.00, -1.57), (-50.00, -150.00, 0.16), None),
    ("can_see_T3_D4_1", (0.00, 0.00, -1.57), (-25.00, -150.00, 2.99), 152.07),
    ("can_see_T3_D4_2", (0.00, 0.00, -1.57), (0.00, -150.00, 4.81), 150),
    ("can_see_T3_D4_3", (0.00, 0.00, -1.57), (25.00, -150.00, 6.19), 152.07),
    ("can_see_T3_D4_4", (0.00, 0.00, -1.57), (50.00, -150.00, 2.51), None),
]


def fmt(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.2f}"


@pytest.mark.parametrize("desc, r1, r2, expected", can_see_testdata)
def test_can_see(desc: str, r1: Tuple, r2: Tuple, expected: float | None):
    rs1 = rw.RobotState(xpos=r1[0], ypos=r1[1], direction=r1[2])
    rs2 = rw.RobotState(xpos=r2[0], ypos=r2[1], direction=r2[2])
    assert fmt(rw.can_see(rs1, rs2)) == fmt(expected)


winner_testdata = [
    ({"r1": [["draw", "true"]], "r2": [["draw", "true"]]}, 0),
    ({"r1": [["winner", "true"]], "r2": []}, 1),
    ({"r1": [], "r2": [["winner", "true"]]}, 2),
]


@pytest.mark.parametrize("obj, expected", winner_testdata)
def test_winner(obj: dict, expected: int):
    result = rwu.parse_sim_winner(obj)
    assert result == expected


collisions_testdata = [
    (
        "wall-rw-6b1d59-00_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=2,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-01_tumblr_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=0,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.PUSH,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=0,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.IS_PUSHED,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-02_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=3,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=2,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-03_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-04_tumblr_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-05_stay-in-field_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=3,
                is_pushed_collision_count=2,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=3,
                is_pushed_collision_count=2,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-06_tumblr_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-07_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=2,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-08_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-09_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-10_tumblr_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=2,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=2,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-11_tumblr_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=1,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.PUSH,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=0,
                is_pushed_collision_count=2,
                end=RobotEventsEnd.IS_PUSHED,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-12_stay-in-field_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=1,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=2,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-13_tumblr_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=2,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.PUSH,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=1,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.IS_PUSHED,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-14_stay-in-field_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-15_tumblr_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.PUSH,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.IS_PUSHED,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-16_stay-in-field_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-17_tumblr_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=2,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.PUSH,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=1,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.IS_PUSHED,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-18_stay-in-field_blind-tumblr.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),
    (
        "wall-rw-6b1d59-19_tumblr_stay-in-field.json",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.WINNER,
                push_collision_count=0,
                is_pushed_collision_count=1,
                end=RobotEventsEnd.PUSH,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.LOOSER,
                push_collision_count=1,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.IS_PUSHED,
            ),
        ),
    ),
]


@pytest.mark.parametrize("file_name, expected", collisions_testdata)
def test_collisions(file_name: str, expected: rw.SimEvents):
    base_dir = Path(__file__).parent / "data"
    file = base_dir / file_name
    sim = rwu.read_file(file)
    result = rw.sim_events(sim)
    assert result == expected


overlap_testdata = [
    (rwh.Interval(0, 10), rwh.Interval(20, 41), False),
    (rwh.Interval(0, 21), rwh.Interval(20, 41), True),
    (rwh.Interval(21, 22), rwh.Interval(20, 41), True),
    (rwh.Interval(40, 42), rwh.Interval(20, 41), True),
    (rwh.Interval(45, 52), rwh.Interval(20, 41), False),
]


@pytest.mark.parametrize("i1, i2, expected", overlap_testdata)
def test_overlap(i1: rwh.Interval, i2: rwh.Interval, expected: bool):
    rwh.validate_interval(i1)
    rwh.validate_interval(i2)
    result = rwh.overlapping(i1, i2)
    assert result == expected


validate_testdata = [
    (rwh.Interval(10, 20), True),
    (rwh.Interval(21, 21), True),
    (rwh.Interval(20, 10), False),
]


@pytest.mark.parametrize("i, is_valid", validate_testdata)
def test_validate(i: rwh.Interval, is_valid: bool):
    if is_valid:
        rwh.validate_interval(i)
    else:
        with pytest.raises(ValueError):
            rwh.validate_interval(i)