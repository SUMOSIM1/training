from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


import math
import pprint as pp
import numpy as np

import training.vector_helper as vh
import training.simrunner_core as src


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass(frozen=True)
class CollisionIntervals:
    inner: list[Interval]
    end: Interval | None


@dataclass(frozen=True)
class Sim:
    states: list[src.SimulationState]
    properties1: list[list]
    properties2: list[list]


class SimWinner(Enum):
    ROBOT1 = "robot1"
    ROBOT2 = "robot2"
    NONE = "none"


class RobotEventsResult(Enum):
    WINNER = "winner"
    LOOSER = "looser"
    DRAW = "draw"


class RobotPushEvents(Enum):
    PUSH = "push"
    IS_PUSHED = "is-pushed"
    NONE = "none"


def intervals_from_threshold(
    values1: list[float], threshold: float, end_len: int
) -> CollisionIntervals:
    """

    :rtype: object
    """
    result = []
    in_interval = False
    start = 0
    for i in range(len(values1) - 1):
        a = values1[i]
        b = values1[i + 1]
        if not in_interval and a > threshold and b <= threshold:
            in_interval = True
            start = i
        elif in_interval and a <= threshold and b > threshold:
            in_interval = False
            result.append(Interval(start, i))
    if in_interval:
        return CollisionIntervals(result, Interval(start, start + end_len))
    return CollisionIntervals(result, None)


def overlapping(a: Interval, b: Interval) -> bool:
    # assume that a and b are valid intervals
    if a.end < b.start:
        return False
    return not a.start > b.end


def validate_interval(i: Interval):
    if i.start > i.end:
        raise ValueError(f"End value of interval must be grater than start value {i}")


def intervals_boolean(booleans: list[bool]) -> list[Interval]:
    in_interval = False
    start = 0
    result = []
    for i in range(len(booleans) - 1):
        a = booleans[i]
        b = booleans[i + 1]
        if not in_interval and not a and b:
            start = i
            in_interval = True
        elif in_interval and a and not b:
            in_interval = False
            result.append(Interval(start, i))
    if in_interval:
        result.append(Interval(start, len(booleans) - 1))
    return result


@dataclass
class CollisionsCount:
    push_count: int
    is_pushed_count: int


def collisions_count(
    collisions: list[Interval], sees: list[Interval]
) -> CollisionsCount:
    def any_match(collision: Interval) -> int:
        for see in sees:
            if overlapping(collision, see):
                return 1
        return 0

    push_count = sum([any_match(coll) for coll in collisions])
    return CollisionsCount(
        push_count=push_count,
        is_pushed_count=len(collisions) - push_count,
    )


def see_intervals(
    states: list[src.SimulationState],
) -> tuple[list[Interval], list[Interval]]:
    r1_see = [bool(can_see(s.robot1, s.robot2)) for s in states]
    r2_see = [bool(can_see(s.robot2, s.robot1)) for s in states]

    r1_see_intervals = intervals_boolean(r1_see)
    r2_see_intervals = intervals_boolean(r2_see)

    return (
        r1_see_intervals,
        r2_see_intervals,
    )


def can_see(robot1: src.PosDir, robot2: src.PosDir) -> float | None:
    def are_clockwise(v1, v2):
        return (-(v1[0] * v2[1]) + (v1[1] * v2[0])) < 0.0

    def is_point_in_sector(v_rel_point, v_start, v_end):
        c1 = are_clockwise(v_rel_point, v_end)
        c2 = are_clockwise(v_start, v_rel_point)
        return c1 and c2

    def inner(alpha: float, r: float) -> bool | None:
        start = robot1.direction - (alpha / 2.0)
        end = robot1.direction + (alpha / 2.0)
        pol_start = np.array([1.0, start])
        pol_end = np.array([1.0, end])

        v_start = vh.pol2cart(pol_start) * 400
        v_end = vh.pol2cart(pol_end) * 400
        v_origin = np.array([robot1.xpos, robot1.ypos])
        v_point = np.array([robot2.xpos, robot2.ypos])
        v_rel_point = v_point - v_origin

        if is_point_in_sector(v_rel_point, v_start, v_end):
            d = np.linalg.norm(v_point - v_origin).item()
            return d if d <= r else None
        return None

    i1 = inner(math.radians(20.0), 400)
    return i1 if i1 else inner(math.radians(80.0), 150)


def end_push_events(
    collision_end_interval: Interval | None,
    r1_see_intervals: list[Interval],
    r2_see_intervals: list[Interval],
    winner: SimWinner,
) -> tuple[RobotPushEvents, RobotPushEvents]:
    if collision_end_interval:
        if winner == SimWinner.ROBOT1:
            robot1_push_out = any(
                overlapping(si, collision_end_interval) for si in r1_see_intervals
            )
            if robot1_push_out:
                return RobotPushEvents.PUSH, RobotPushEvents.IS_PUSHED
            return RobotPushEvents.NONE, RobotPushEvents.NONE
        elif winner == SimWinner.ROBOT2:
            robot2_push_out = any(
                overlapping(si1, collision_end_interval) for si1 in r2_see_intervals
            )
            if robot2_push_out:
                return RobotPushEvents.IS_PUSHED, RobotPushEvents.PUSH
            return RobotPushEvents.NONE, RobotPushEvents.NONE
        return RobotPushEvents.NONE, RobotPushEvents.NONE
    return RobotPushEvents.NONE, RobotPushEvents.NONE


def dist(state: src.SimulationState) -> float:
    r1 = state.robot1
    r2 = state.robot2
    dx = r1.xpos - r2.xpos
    dy = r1.ypos - r2.ypos
    return math.sqrt(dx * dx + dy * dy)


def parse_robo_properties(
    properties1: list[list[(str, str)]], properties2: list[list[(str, str)]]
) -> RobotEventsResult:
    """
    :param properties1:
        Key value pairs
    :param properties2:
        Key value pairs
    :return:
        ['draw', 'true'] => DRAW
        ['winner', 'true'] => WINNER
        else => LOOSER
    """
    valid = validate_properties(properties1, properties2)
    if valid is not None:
        raise ValueError(valid)
    result = RobotEventsResult.LOOSER
    if properties1 and properties1[0]:
        _dict = dict(properties1)
        if _dict.get("draw") == "true":
            result = RobotEventsResult.DRAW
        elif _dict.get("winner") == "true":
            result = RobotEventsResult.WINNER
    return result


def validate_properties(
    properties_robot1: list[list[(str, str)]], properties_robot2: list[list[(str, str)]]
) -> str | None:
    def has_property(properties: list[list[(str, str)]], key: str, value: str) -> bool:
        filtered = [prop for prop in properties if prop[0] == key and prop[1] == value]
        re = len(filtered) > 0
        return re

    def f() -> str:
        f1 = pp.pformat(properties_robot1)
        f2 = pp.pformat(properties_robot2)
        return f"1:{f1} 2:{f2}"

    def any_invalid_prop(props: list[list[(str, str)]]) -> str | None:
        def is_valid(_props: list[str]) -> str | None:
            if len(_props) != 2:
                return f"property {_props} is not valid"
            return None

        for prop in props:
            valid = is_valid(prop)
            if valid is not None:
                return valid
        return None

    v1 = any_invalid_prop(properties_robot1)
    if v1 is not None:
        return v1

    v2 = any_invalid_prop(properties_robot2)
    if v2 is not None:
        return v2

    if has_property(properties_robot1, "draw", "true"):  # noqa: SIM102
        if not has_property(properties_robot2, "draw", "true"):
            return f"If 'draw', the opponent also has to be 'draw' {f()}"
    if has_property(properties_robot1, "winner", "true"):
        if has_property(properties_robot2, "draw", "true"):
            return f"If 'winner', the opponent cannot be 'draw' {f()}"
        if has_property(properties_robot2, "winner", "true"):
            return f"If 'winner', the opponent cannot be 'winner' {f()}"
    if has_property(properties_robot2, "winner", "true"):
        if has_property(properties_robot1, "draw", "true"):
            return f"If 'winner', the opponent cannot be 'draw' {f()}"
        if has_property(properties_robot1, "winner", "true"):
            return f"If 'winner', the opponent cannot be 'winner' {f()}"


def parse_sim_winner(properties_robot1: list[list[(str, str)]]) -> SimWinner:
    """
    :param properties_robot1: dict containing winner info
        ['draw', 'true'] => NONE
        ['winner', 'true'] => ROBOT1
        else => ROBOT2
    :return: SimWinner
    """
    if properties_robot1:
        if properties_robot1[0][0] == "draw":
            return SimWinner.NONE
        elif properties_robot1[0][0] == "winner":
            return SimWinner.ROBOT1
    return SimWinner.ROBOT2


def read_sim_from_file(file: Path) -> Sim:
    with file.open() as f:
        data_dict = json.load(f)
        states_object = data_dict["states"]
        states = [src.SimulationState.from_dict(s) for s in states_object]
        properties1 = data_dict["winner"]["r1"]
        properties2 = data_dict["winner"]["r2"]
        return Sim(states, properties1, properties2)
