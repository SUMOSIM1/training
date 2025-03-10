from abc import ABC, abstractmethod
from dataclasses import dataclass


import training.consts as consts
import training.reward.reward_helper as rh
import training.reward.reward_core as rhc
import training.simrunner_core as src


@dataclass
class RobotContinuousEndEvents:
    result: rh.RobotEventsResult
    end: rh.RobotPushEvents
    # A value between 0 and one relative to MAX_STEPS
    steps_count_relative: float


@dataclass
class RobotContinuousEvents:
    robot_push_events: rh.RobotPushEvents


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
            case rh.RobotPushEvents.PUSH:
                # print("----- map_robot_continuous_events -----")
                # pp.pprint(events)
                return self.properties().push_reward
            case rh.RobotPushEvents.IS_PUSHED:
                # print("----- map_robot_continuous_events -----")
                # pp.pprint(events)
                return self.properties().is_pushed_penalty
            case rh.RobotPushEvents.NONE:
                return 0.0
            case _:
                raise ValueError(f"Unknown RobotPushEvents;{events.robot_push_events}")

    def map_robot_continuous_end_events(
        self, events: RobotContinuousEndEvents
    ) -> float:
        # print("----- map_robot_continuous_end_events -----")
        # pp.pprint(events)
        match events.result:
            case rh.RobotEventsResult.WINNER:
                match events.end:
                    case rh.RobotPushEvents.PUSH:
                        # Return the highest possible reward extras for fast winning
                        return (
                            self.properties().win_by_push_reward
                            + self.win_by_push_duratin_reward(events)
                        )
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
                        return (
                            self.properties().loose_unforced_penalty
                            + self.loose_unforced_duration_penalty(events)
                        )
                    case rh.RobotPushEvents.IS_PUSHED:
                        return self.properties().loose_forced_penalty
                    case _:
                        raise ValueError(f"Unknown RobotPushEvents:{events.end}")
            case _:
                raise ValueError(f"Unknown RobotEventsResult:{events}")


class ContinuousRewardHandler(rhc.RewardHandler):
    @abstractmethod
    def event_mapper(self) -> EventMapper:
        pass

    def calculate_reward(self, state: src.SimulationState) -> tuple[float, float]:
        r1_events, r2_events = continuous_events_from_simulation_state(state)
        reward1 = self.event_mapper().map_robot_continuous_events(r1_events)
        reward2 = self.event_mapper().map_robot_continuous_events(r2_events)
        return reward1, reward2

    def calculate_end_reward(
        self,
        states: list[src.SimulationState],
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
        return rhc.RewardHandlerName.CONTINUOUS_CONSIDER_ALL.value


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
        return rhc.RewardHandlerName.REDUCED_PUSH_REWARD.value


class SpeedBonusRewardHandler(ContinuousRewardHandler):
    class Mapper(AbstractEventMapper):
        mapper_props = AbstractEventMapperPoperties(
            win_by_push_reward=100.0,
            max_win_by_push_duration_reward=150.0,
            loose_unforced_penalty=-100.0,
            max_loose_unforced_duration_penalty=-150.0,
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
        return rhc.RewardHandlerName.SPEED_BONUS.value


def continuous_end_events_from_simulation_states(
    states: list[src.SimulationState],
    properties1: list[list[(str, str)]],
    properties2: list[list[(str, str)]],
    simulation_max_steps: int,
) -> tuple[RobotContinuousEndEvents, RobotContinuousEndEvents]:
    r1_result = rh.parse_robo_properties(properties1, properties2)
    r2_result = rh.parse_robo_properties(properties2, properties1)

    r1_see_intervals, r2_see_intervals = rh.see_intervals(states)

    distances = list([rh.dist(s) for s in states])
    collision_intervals: rh.CollisionIntervals = rh.intervals_from_threshold(
        distances, consts.ROBOT_DIAMETER + 5, int(consts.FIELD_DIAMETER)
    )
    r1_end, r2_end = rh.end_push_events(
        collision_intervals.end,
        r1_see_intervals,
        r2_see_intervals,
        rh.parse_sim_winner(properties1),
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
    state: src.SimulationState,
) -> tuple[RobotContinuousEvents, RobotContinuousEvents]:
    def push_events(
        dist: float, _can_see: bool, other_can_see: bool
    ) -> rh.RobotPushEvents:
        if dist < consts.ROBOT_DIAMETER + 5:
            if _can_see:
                return rh.RobotPushEvents.PUSH
            if other_can_see:
                return rh.RobotPushEvents.IS_PUSHED
        return rh.RobotPushEvents.NONE

    _dist = rh.dist(state)
    robot1_can_see = rh.can_see(state.robot1, state.robot2)
    robot2_can_see = rh.can_see(state.robot2, state.robot1)
    robot1_push_events = push_events(_dist, bool(robot1_can_see), bool(robot2_can_see))
    robot2_push_events = push_events(_dist, bool(robot2_can_see), bool(robot1_can_see))
    e1 = RobotContinuousEvents(robot_push_events=robot1_push_events)
    e2 = RobotContinuousEvents(robot_push_events=robot2_push_events)
    return e1, e2
