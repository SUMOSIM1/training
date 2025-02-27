import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pprint as pp

import numpy as np

import training.consts as consts
import training.reward.reward_helper as rh
import training.simrunner as sr
import training.vector_helper as vh


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


@dataclass
class RobotContinuousEndEvents:
    result: RobotEventsResult
    end: RobotPushEvents
    # A value between 0 and one relative to MAX_STEPS
    steps_count_relative: float


@dataclass
class RobotContinuousEvents:
    robot_push_events: RobotPushEvents


class EventMapper(ABC):
    @abstractmethod
    def map_robot_continuous_events(self, events: RobotContinuousEvents) -> float:
        pass

    @abstractmethod
    def map_robot_continuous_end_events(
        self, events: RobotContinuousEndEvents
    ) -> float:
        pass


@dataclass
class AbstractEventMapperPoperties:
    win_by_push_reward: float
    max_win_by_push_duration_reward: float
    loose_unforced_penalty: float
    max_loose_unforced_duration_penalty: float
    loose_forced_penalty: float
    push_reward: float
    is_pushed_penalty: float


class AbstractEventMapper(EventMapper):
    @abstractmethod
    def properties(self) -> AbstractEventMapperPoperties:
        pass

    def win_by_push_duratin_reward(self, events: RobotContinuousEndEvents) -> float:
        r = (
            1.0 - events.steps_count_relative
        ) * self.properties().max_win_by_push_duration_reward
        # print(f"---- winning {r}")
        return r

    def loose_unforced_duration_penalty(
        self, events: RobotContinuousEndEvents
    ) -> float:
        r = (
            1.0 - events.steps_count_relative
        ) * self.properties().max_loose_unforced_duration_penalty
        # print(f"---- loosing {r}")
        return r

    def map_robot_continuous_events(self, events: RobotContinuousEvents) -> float:
        match events.robot_push_events:
            case RobotPushEvents.PUSH:
                # print("----- map_robot_continuous_events -----")
                # pp.pprint(events)
                return self.properties().push_reward
            case RobotPushEvents.IS_PUSHED:
                # print("----- map_robot_continuous_events -----")
                # pp.pprint(events)
                return self.properties().is_pushed_penalty
            case RobotPushEvents.NONE:
                return 0.0
            case _:
                raise ValueError(f"Unknown RobotPushEvents;{events.robot_push_events}")

    def map_robot_continuous_end_events(
        self, events: RobotContinuousEndEvents
    ) -> float:
        # print("----- map_robot_continuous_end_events -----")
        # pp.pprint(events)
        match events.result:
            case RobotEventsResult.WINNER:
                match events.end:
                    case RobotPushEvents.PUSH:
                        # Return the highest possible reward extras for fast winning
                        return (
                            self.properties().win_by_push_reward
                            + self.win_by_push_duratin_reward(events)
                        )
                    case RobotPushEvents.NONE:
                        return 0.0
                    case RobotPushEvents.IS_PUSHED:
                        raise ValueError(
                            f"Unexpected combination: result:{events.result} "
                            f"and end:{events.end}"
                        )
                    case _:
                        raise ValueError(f"Unknown RobotPushEvents:{events.end}")
            case RobotEventsResult.DRAW:
                return 0.0
            case RobotEventsResult.LOOSER:
                match events.end:
                    case RobotPushEvents.PUSH:
                        raise ValueError(
                            f"Unexpected combination: result:{events.result} "
                            f"and end:{events.end}"
                        )
                    case RobotPushEvents.NONE:
                        # Running unforced out of the field is the worst you can do.
                        # The penalty is higher if you leave the field earlier
                        return (
                            self.properties().loose_unforced_penalty
                            + self.loose_unforced_duration_penalty(events)
                        )
                    case RobotPushEvents.IS_PUSHED:
                        return self.properties().loose_forced_penalty
                    case _:
                        raise ValueError(f"Unknown RobotPushEvents:{events.end}")
            case _:
                raise ValueError(f"Unknown RobotEventsResult:{events}")


class ContinuousRewardHandler(sr.RewardHandler):
    @abstractmethod
    def event_mapper(self) -> EventMapper:
        pass

    def calculate_reward(self, state: sr.SimulationState) -> tuple[float, float]:
        r1_events, r2_events = continuous_events_from_simulation_state(state)
        reward1 = self.event_mapper().map_robot_continuous_events(r1_events)
        reward2 = self.event_mapper().map_robot_continuous_events(r2_events)
        return reward1, reward2

    def calculate_end_reward(
        self,
        states: list[sr.SimulationState],
        properties1: list[list],
        properties2: list[list],
        max_simulation_steps: int,
    ) -> tuple[float, float]:
        r1_events, r2_events = continuous_end_events_from_simulation_states(
            states, properties1, properties2, max_simulation_steps
        )
        reward1 = self.event_mapper().map_robot_continuous_end_events(r1_events)
        reward2 = self.event_mapper().map_robot_continuous_end_events(r2_events)
        return reward1, reward2


class ConsiderAllRewardHandler(ContinuousRewardHandler):
    class Mapper(AbstractEventMapper):
        mapper_props = AbstractEventMapperPoperties(
            win_by_push_reward=100.0,
            max_win_by_push_duration_reward=50.0,
            loose_unforced_penalty=-100.0,
            max_loose_unforced_duration_penalty=-50.0,
            loose_forced_penalty=-10.0,
            push_reward=0.5,
            is_pushed_penalty=-0.1,
        )

        def properties(self):
            return self.mapper_props

    def __init__(self):
        self.em: EventMapper = self.Mapper()

    def event_mapper(self) -> EventMapper:
        return self.em

    def name(self) -> str:
        return sr.RewardHandlerName.CONTINUOUS_CONSIDER_ALL.value


class ReducedPushRewardHandler(ContinuousRewardHandler):
    class Mapper(AbstractEventMapper):
        mapper_props = AbstractEventMapperPoperties(
            win_by_push_reward=100.0,
            max_win_by_push_duration_reward=50.0,
            loose_unforced_penalty=-100.0,
            max_loose_unforced_duration_penalty=-50.0,
            loose_forced_penalty=-10.0,
            push_reward=0.05,
            is_pushed_penalty=-0.01,
        )

        def properties(self):
            return self.mapper_props

    def __init__(self):
        self.em: EventMapper = self.Mapper()

    def event_mapper(self) -> EventMapper:
        return self.em

    def name(self) -> str:
        return sr.RewardHandlerName.REDUCED_PUSH_REWARD.value


def continuous_end_events_from_simulation_states(
    states: list[sr.SimulationState],
    properties1: list[list[(str, str)]],
    properties2: list[list[(str, str)]],
    simulation_max_steps: int,
) -> tuple[RobotContinuousEndEvents, RobotContinuousEndEvents]:
    r1_result = _parse_robo_properties(properties1, properties2)
    r2_result = _parse_robo_properties(properties2, properties1)

    r1_see_intervals, r2_see_intervals = see_intervals(states)

    distances = list([dist(s) for s in states])
    collision_intervals: rh.CollisionIntervals = rh.intervals_from_threshold(
        distances, consts.ROBOT_DIAMETER + 5, int(consts.FIELD_DIAMETER)
    )
    r1_end, r2_end = end_push_events(
        collision_intervals.end,
        r1_see_intervals,
        r2_see_intervals,
        _parse_sim_winner(properties1),
    )
    steps_count_relative = float(len(states)) / simulation_max_steps

    e1 = RobotContinuousEndEvents(
        result=r1_result,
        end=r1_end,
        steps_count_relative=steps_count_relative,
    )
    e2 = RobotContinuousEndEvents(
        result=r2_result,
        end=r2_end,
        steps_count_relative=steps_count_relative,
    )
    return e1, e2


def continuous_events_from_simulation_state(
    state: sr.SimulationState,
) -> tuple[RobotContinuousEndEvents, RobotContinuousEndEvents]:
    def push_events(dist: float, can_see: bool, other_can_see: bool) -> RobotPushEvents:
        if dist < consts.ROBOT_DIAMETER + 5:
            if can_see:
                return RobotPushEvents.PUSH
            if other_can_see:
                return RobotPushEvents.IS_PUSHED
        return RobotPushEvents.NONE

    _dist = dist(state)
    robot1_can_see = can_see(state.robot1, state.robot2)
    robot2_can_see = can_see(state.robot2, state.robot1)
    robot1_push_events = push_events(_dist, bool(robot1_can_see), bool(robot2_can_see))
    robot2_push_events = push_events(_dist, bool(robot2_can_see), bool(robot1_can_see))
    e1 = RobotContinuousEvents(robot_push_events=robot1_push_events)
    e2 = RobotContinuousEvents(robot_push_events=robot2_push_events)
    return e1, e2


def see_intervals(
    states: list[sr.SimulationState],
) -> tuple[list[rh.Interval], list[rh.Interval]]:
    r1_see = [bool(can_see(s.robot1, s.robot2)) for s in states]
    r2_see = [bool(can_see(s.robot2, s.robot1)) for s in states]

    r1_see_intervals = rh.intervals_boolean(r1_see)
    r2_see_intervals = rh.intervals_boolean(r2_see)

    return (
        r1_see_intervals,
        r2_see_intervals,
    )


def can_see(robot1: sr.PosDir, robot2: sr.PosDir) -> float | None:
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
    collision_end_interval: rh.Interval | None,
    r1_see_intervals: list[rh.Interval],
    r2_see_intervals: list[rh.Interval],
    winner: SimWinner,
) -> tuple[RobotPushEvents, RobotPushEvents]:
    def is_winner_push(
        end_interval: rh.Interval, see_intervals: list[rh.Interval]
    ) -> bool:
        return any(rh.overlapping(si, end_interval) for si in see_intervals)

    if collision_end_interval:
        if winner == SimWinner.ROBOT1:
            robot1_push_out = any(
                rh.overlapping(si, collision_end_interval) for si in r1_see_intervals
            )
            if robot1_push_out:
                return RobotPushEvents.PUSH, RobotPushEvents.IS_PUSHED
            return RobotPushEvents.NONE, RobotPushEvents.NONE
        elif winner == SimWinner.ROBOT2:
            robot2_push_out = any(
                rh.overlapping(si1, collision_end_interval) for si1 in r2_see_intervals
            )
            if robot2_push_out:
                return RobotPushEvents.IS_PUSHED, RobotPushEvents.PUSH
            return RobotPushEvents.NONE, RobotPushEvents.NONE
        return RobotPushEvents.NONE, RobotPushEvents.NONE
    return RobotPushEvents.NONE, RobotPushEvents.NONE


def dist(state: sr.SimulationState) -> float:
    r1 = state.robot1
    r2 = state.robot2
    dx = r1.xpos - r2.xpos
    dy = r1.ypos - r2.ypos
    return math.sqrt(dx * dx + dy * dy)


def _parse_robo_properties(
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
    valid = _validate_properties(properties1, properties2)
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


def _validate_properties(
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
        def is_valid(p: list[str]) -> str | None:
            if len(p) != 2:
                return f"property {p} is not valid"
            return None

        for p in props:
            v = is_valid(p)
            if v is not None:
                return v
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


def _parse_sim_winner(properties_robot1: list[list[(str, str)]]) -> SimWinner:
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
