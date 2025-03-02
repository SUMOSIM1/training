"""
Currently not used. Contains the implementation of end only reward handler.
this might be usable for other learning methods.

To be refactored. Remove the continuous reward handler that is now in reward1

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


import training.consts as consts
import training.reward.reward_helper as rh
import training.simrunner as sr


@dataclass
class RobotContinuousEndEvents:
    result: rh.RobotEventsResult
    end: rh.RobotPushEvents
    # A value between 0 and one relative to MAX_STEPS
    steps_count_relative: float


@dataclass
class RobotEndEvents(RobotContinuousEndEvents):
    push_collision_count: int
    is_pushed_collision_count: int


@dataclass
class RobotContinuousEvents:
    robot_push_events: rh.RobotPushEvents


class EventMapper(ABC):
    @abstractmethod
    def map_robot_continuous_events(self, events: RobotContinuousEvents) -> float:
        pass

    @abstractmethod
    def map_robot_end_events(self, events: RobotEndEvents) -> float:
        pass

    @abstractmethod
    def map_robot_continuous_end_events(
        self, events: RobotContinuousEndEvents
    ) -> float:
        pass


class ConsiderAllEventMapper(EventMapper):
    def map_robot_end_events(self, events: RobotEndEvents) -> float:
        return 0.0

    def map_robot_continuous_events(self, events: RobotContinuousEvents) -> float:
        match events.robot_push_events:
            case rh.RobotPushEvents.PUSH:
                return 0.5
            case rh.RobotPushEvents.IS_PUSHED:
                return -0.1
            case rh.RobotPushEvents.NONE:
                return 0.0
            case _:
                raise ValueError(f"Unknown RobotPushEvents;{events.robot_push_events}")

    def map_robot_continuous_end_events(
        self, events: RobotContinuousEndEvents
    ) -> float:
        match events.result:
            case rh.RobotEventsResult.WINNER:
                match events.end:
                    case rh.RobotPushEvents.PUSH:
                        # Return the highest possible reward extras for fast winning
                        return 100.0 + fast_winning_reward(events)
                    case rh.RobotPushEvents.NONE:
                        return 0.0
                    case rh.RobotPushEvents.IS_PUSHED:
                        raise ValueError(
                            f"Unexpected combination: result:{events.result} "
                            f"and end:{events.end}"
                        )
                    case _:
                        raise ValueError(f"Unknown RobotPushEvents:{events.end}")
            case rh.RobotEventsResult.DRAW:
                return 0.0
            case rh.RobotEventsResult.LOOSER:
                match events.end:
                    case rh.RobotPushEvents.PUSH:
                        raise ValueError(
                            f"Unexpected combination: result:{events.result} "
                            f"and end:{events.end}"
                        )
                    case rh.RobotPushEvents.NONE:
                        # Running unforced out of the field is the worst you can do.
                        # The penalty is higher if you leave the field earlier
                        return -100.0 + fast_loosing_penalty(events)
                    case rh.RobotPushEvents.IS_PUSHED:
                        return -10.0
                    case _:
                        raise ValueError(f"Unknown RobotPushEvents:{events.end}")
            case _:
                raise ValueError(f"Unknown RobotEventsResult:{events}")


def map_robot_end_events(self, events: RobotEndEvents) -> float:
    match events.result:
        case rh.RobotEventsResult.WINNER:
            match events.end:
                case rh.RobotPushEvents.PUSH:
                    # Return the highest possible reward extras for fast winning
                    return 100.0 + fast_winning_reward(events)
                case rh.RobotPushEvents.NONE:
                    # You win because your opponent left the field.
                    # You did not push him.
                    # Just count the pushes and is_pushed like on draw
                    return is_pushed_penalty(events) + pushing_reward(events)
                case rh.RobotPushEvents.IS_PUSHED:
                    raise ValueError(
                        f"Unexpected combination: result:{events.result} "
                        f"and end:{events.end}"
                    )
                case _:
                    raise ValueError(f"Unknown RobotPushEvents:{events.end}")
        case rh.RobotEventsResult.DRAW:
            # Just count the pushes and is_pushed. To push is higher rated
            # than the being pushed penalty
            return self.is_pushed_penalty(events) + pushing_reward(events)
        case rh.RobotEventsResult.LOOSER:
            match events.end:
                case rh.RobotPushEvents.PUSH:
                    raise ValueError(
                        f"Unexpected combination: result:{events.result} "
                        f"and end:{events.end}"
                    )
                case rh.RobotPushEvents.NONE:
                    # Running unforced out of the field is the worst you can do.
                    # The penalty is higher if you leave the field earlier
                    return -100.0 + self.fast_loosing_penalty(events)
                case rh.RobotPushEvents.IS_PUSHED:
                    # You get a moderate penalty for being pushed out.
                    # How you behaved during the match is taken in account
                    return -10.0 + is_pushed_penalty(events) + pushing_reward(events)
                case _:
                    raise ValueError(f"Unknown RobotPushEvents:{events.end}")
        case _:
            raise ValueError(f"Unknown RobotEventsResult:{events}")


def fast_winning_reward(events: RobotContinuousEndEvents) -> float:
    return (1.0 - events.steps_count_relative) * 50


def fast_loosing_penalty(events: RobotContinuousEndEvents) -> float:
    return (1.0 - events.steps_count_relative) * -50


def pushing_reward(events: RobotEndEvents) -> float:
    return events.push_collision_count * 10.0


def is_pushed_penalty(events: RobotEndEvents) -> float:
    return events.is_pushed_collision_count * -2.0


class EndRewardHandler(sr.RewardHandler):
    @abstractmethod
    def event_mapper(self) -> EventMapper:
        pass

    def calculate_reward(self, state: sr.SimulationState) -> tuple[float, float]:
        return 0.0, 0.0

    def calculate_end_reward(
        self,
        states: list[sr.SimulationState],
        properties1: list[list],
        properties2: list[list],
        max_simulation_steps: int,
    ) -> tuple[float, float]:
        r1_events, r2_events = end_events_from_simulation_states(
            states, properties1, properties2, max_simulation_steps
        )
        reward1 = self.event_mapper().map_robot_end_events(r1_events)
        reward2 = self.event_mapper().map_robot_end_events(r2_events)
        return reward1, reward2


class EndConsiderAllRewardHandler(EndRewardHandler):
    def __init__(self):
        self.em: EventMapper = ConsiderAllEventMapper()

    def name(self) -> str:
        return sr.RewardHandlerName.END_CONSIDER_ALL.value

    def event_mapper(self) -> EventMapper:
        return self.em


def end_events_from_simulation_states(
    states: list[sr.SimulationState],
    properties1: list[list[(str, str)]],
    properties2: list[list[(str, str)]],
    max_simulation_steps: int,
) -> tuple[RobotEndEvents, RobotEndEvents]:
    """
    Collect the events for both robots during the match
    :param max_simulation_steps: Maximum number of steps
    :param states: All states of the match
    :param properties1: Properties collected for robot 1
    :param properties2: Properties collected for robot 2
    :return: Events for robot1 and robot2
    """

    r1_result = rh.parse_robo_properties(properties1, properties2)
    r2_result = rh.parse_robo_properties(properties2, properties1)

    distances = list([rh.dist(s) for s in states])
    collision_intervals: rh.CollisionIntervals = rh.intervals_from_threshold(
        distances, consts.ROBOT_DIAMETER + 5, int(consts.FIELD_DIAMETER)
    )
    r1_see_intervals, r2_see_intervals = rh.see_intervals(states)
    coll_count1 = rh.collisions_count(collision_intervals.inner, r1_see_intervals)
    coll_count2 = rh.collisions_count(collision_intervals.inner, r2_see_intervals)

    r1_end, r2_end = rh.end_push_events(
        collision_intervals.end,
        r1_see_intervals,
        r2_see_intervals,
        rh.parse_sim_winner(properties1),
    )
    steps_count_relative = float(len(states)) / max_simulation_steps

    e1 = RobotEndEvents(
        result=r1_result,
        push_collision_count=coll_count1.push_count,
        is_pushed_collision_count=coll_count1.is_pushed_count,
        end=r1_end,
        steps_count_relative=steps_count_relative,
    )
    e2 = RobotEndEvents(
        result=r2_result,
        push_collision_count=coll_count2.push_count,
        is_pushed_collision_count=coll_count2.is_pushed_count,
        end=r2_end,
        steps_count_relative=steps_count_relative,
    )
    return e1, e2
