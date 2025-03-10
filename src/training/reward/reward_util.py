import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import training.reward.reward0 as rw0
import training.reward.reward_helper as rwh
import training.simdb as sd
import training.simrunner_core as src


@dataclass
class Sim:
    states: list[src.SimulationState]
    properties1: list[list]
    properties2: list[list]


def util_sims_to_json_files(
    sim_names: list[str], base_dir: Path, db_host: str, db_port: int
) -> list[Path]:
    out_files = []
    with sd.create_client(db_host, db_port) as c:
        all_sims = sd.find_all(c)
        for sim in all_sims:
            if sim["name"] in sim_names:
                of = db_sim_to_json_file(sim, base_dir)
                out_files.append(of)
    return out_files


def db_sim_to_json_file(db_sim: dict, base_dir: Path) -> Path:
    name = db_sim["name"]
    robot_name1 = db_sim["robot1"]["name"]
    robot_name2 = db_sim["robot2"]["name"]
    sim_name = f"{name}_{robot_name1}_{robot_name2}"
    file_name = f"{sim_name}.json"
    file_path = base_dir / file_name
    with file_path.open("w", encoding="UTF-8") as f:
        sim_dict = {
            "name": sim_name,
            "winner": db_sim["events"],
            "states": db_sim["states"],
        }
        json.dump(sim_dict, cast(f, any), indent=2)
    return file_path


def can_see_200(r1: src.PosDir, r2: src.PosDir, h: float) -> float | None:
    v = rwh.can_see(r1, r2)
    if v is not None and v < 200:
        return h
    return None


def print_distances(file: Path):
    sim = read_sim_from_file(file)
    dists = [rwh.dist(s) for s in sim.states]
    for d in dists:
        if d < 101:
            print(d)


def read_sim_from_file(file: Path) -> Sim:
    with file.open() as f:
        data_dict = json.load(f)
        states_object = data_dict["states"]
        states = [src.SimulationState.from_dict(s) for s in states_object]
        properties1 = data_dict["winner"]["r1"]
        properties2 = data_dict["winner"]["r2"]
        return Sim(states, properties1, properties2)


def reward_analysis():
    @dataclass
    class RoboEventsReward:
        robo_events: rw0.RobotEndEvents
        reward: float

    def read_events_reward(file: Path) -> (RoboEventsReward, RoboEventsReward):
        sim = read_sim_from_file(file)
        sim_events1, sim_events2 = rw0.end_events_from_simulation_states(
            sim.states, sim.properties1, sim.properties2, 1000
        )
        rew1, rew2 = reward_handler.calculate_end_reward(
            sim.states, sim.properties1, sim.properties2, 1000
        )
        rer1 = RoboEventsReward(sim_events1, rew1)
        rer2 = RoboEventsReward(sim_events2, rew2)
        return rer1, rer2

    def print_header():
        print()
        print(
            f"{'name':10}"
            f"{'push':>10}"
            f"{'is_pushed':>15}"
            f"{'end':>15}"
            f"{'steps':>10}"
            f"{'reward':>10}"
        )
        print("-" * 70)

    def print_events_reward(events_reward: RoboEventsReward):
        # print(pformat(events_reward))
        print(
            f"{events_reward.robo_events.result.name:10}"
            f"{events_reward.robo_events.push_collision_count:10d}"
            f"{events_reward.robo_events.is_pushed_collision_count:15d}"
            f"{events_reward.robo_events.end.name:>15}"
            f"{events_reward.robo_events.steps_count_relative:10.3f}"
            f"{events_reward.reward:10.3f}"
        )

    def print_result(
        _event_rewards: list[RoboEventsReward], result: rwh.RobotEventsResult
    ):
        filtered = [e for e in _event_rewards if e.robo_events.result == result]
        for er in filtered:
            print_events_reward(er)

    reward_handler = rw0.EndConsiderAllRewardHandler()

    good_bad = """
    What is good or bad for
    
    WINNER:
    end_push + reward for fast winning -> Highest reward
    end_none -> No reward. The opponent waked out
    
    LOOSER:
    end_was_pushed -> Low minus. Do not blame too much a good robot for being pushed out 
        (eventually same reward/penalty for push/is_pushed as for draw)
    end_none + penalty for fast walk out -> High penalty 
        (eventually same reward/penalty for push/is_pushed as for draw)
    
    DRAW:
    reward per push
    penalty per is_pushed (think about how high the reward/penalty should be)
    
    Think about two situations:
    - Trained robot is bad compared to the opponent
    - Trained robot is almost as good as trained robot
    
    """  # noqa: E501

    data_dir = Path(__file__).parent.parent / "tests" / "data"
    files = [file for file in data_dir.iterdir() if file.suffix == ".json"]
    event_rewards = [
        read_events_reward(file) for file in files if file.suffix == ".json"
    ]
    flat_event_rewards = list(sum(event_rewards, ()))
    print(good_bad)

    print_header()
    print_result(flat_event_rewards, rwh.RobotEventsResult.WINNER)
    print_header()
    print_result(flat_event_rewards, rwh.RobotEventsResult.LOOSER)
    print_header()
    print_result(flat_event_rewards, rwh.RobotEventsResult.DRAW)
