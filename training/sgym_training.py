import math
import time
from enum import Enum
from pathlib import Path

import gymnasium as gym
import gymnasium.spaces as gyms
import numpy as np

import training.sgym as sgym
import training.simrunner as sr
import training.sumosim_helper as sh


def cont_act_space(config: sgym.SEnvConfig) -> gym.Space:
    return gyms.Box(
        low=-config.max_wheel_speed,
        high=config.max_wheel_speed,
        shape=(1, 2),
        dtype=config.dtype,
    )


def cont_obs_space(config: sgym.SEnvConfig) -> gym.Space:
    observation_view_space = gyms.Discrete(n=4)
    observation_border_space = gyms.Box(
        low=0.0, high=config.max_view_distance, shape=(1, 3), dtype=config.dtype
    )
    return gyms.Dict(
        {
            "view": observation_view_space,
            "border": observation_border_space,
        }
    )


def map_cont_sensor_to_obs(
    sensor: sr.CombiSensor, config: sgym.SEnvConfig
) -> dict[str, any]:
    def view_mapping() -> int:
        match sensor.opponent_in_sector:
            case sr.SectorName.UNDEF:
                return 0
            case sr.SectorName.LEFT:
                return 1
            case sr.SectorName.CENTER:
                return 2
            case sr.SectorName.RIGHT:
                return 3
            case _:
                raise ValueError(f"Wrong sector name {sensor.opponent_in_sector}")

    return {
        "view": view_mapping(),
        "border": _create_numpy_array(
            [
                [
                    sensor.left_distance,
                    sensor.front_distance,
                    sensor.right_distance,
                ]
            ],
            config,
        ),
    }


def map_cont_act_to_diff_drive(action_space: list[list]) -> sr.DiffDriveValues:
    # TODO check if that mapping is OK
    return sr.DiffDriveValues(
        left_velo=action_space[0][1],
        right_velo=action_space[0][0],
    )


cont_sgym_mapping = sgym.SEnvMapping(
    act_space=cont_act_space,
    obs_space=cont_obs_space,
    map_act=map_cont_act_to_diff_drive,
    map_sensor=map_cont_sensor_to_obs,
)


class SGymLoop(Enum):
    SAMPLE = "sample"
    Q_SAMPLE = "q-sample"


def main(
    sgym_loop: SGymLoop,
    epoch_count: int,
    record: bool,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    match sgym_loop:
        case SGymLoop.SAMPLE:
            sample(epoch_count, record, port, opponent_name, reward_handler_name)
        case SGymLoop.Q_SAMPLE:
            q_sample(epoch_count, record, port, opponent_name, reward_handler_name)


def sample(
    epoch_count: int,
    record: bool,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    print(
        f"### sgym sample e:{epoch_count} p:{port} "
        f"o:{opponent_name.value} rh:{reward_handler_name.value} r:{record}"
    )

    run_id = _tid()

    rewards = []
    training_name = f"SGYM-001-{run_id}"
    for n in range(epoch_count):
        sim_name = f"{training_name}-{n:03d}"
        opponent = sr.ControllerProvider.get(opponent_name)

        sim_info = None
        if record:
            sim_info = sr.SimInfo(
                name1="sample-agent",
                desc1={"info": "Agent with sample actions"},
                name2=opponent.name(),
                desc2=opponent.description(),
                port=port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )

        env = sgym.SEnv(
            senv_config=sgym.default_senv_config,
            senv_mapping=cont_sgym_mapping,
            port=port,
            sim_name=sim_name,
            opponent=opponent,
            reward_handler=reward_handler,
            sim_info=sim_info,
        )
        _observation, _info = env.reset()
        cnt = 0
        episode_over = False
        cuml_reward = 0.0
        while not episode_over:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            cuml_reward += reward
            # if cnt > 0 and cnt % 50 == 0:
            #     print(f"### {cnt} reward:{reward} info:{info}")
            episode_over = terminated or truncated
            cnt += 1

        print(
            f"### finished epoch {sim_name} "
            f"reward:{cuml_reward:10.2f} record:{record}"
        )
        rewards.append(cuml_reward)
        env.close()

    out_dir = Path.home() / "tmp" / "sumosim"
    out_dir.mkdir(exist_ok=True, parents=True)

    desc = {
        "loop name": "sample",
        "reward handler": reward_handler_name.value,
        "epoch count": epoch_count,
    }
    lines = sh.create_lines(desc, [[0], [1, 2]])
    sh.boxplot(rewards, out_dir, training_name, lines)
    print(f"Wrote plot {training_name} to {out_dir}")


def q_sample(
    epoch_count: int,
    record: bool,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    loop_name = "q-sample"
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    print(
        f"### sgym {loop_name} e:{epoch_count} p:{port} "
        f"o:{opponent_name.value} rh:{reward_handler_name.value} r:{record}"
    )

    run_id = _tid()

    rewards = []
    training_name = f"QSAMPLE-001-{run_id}"
    for n in range(epoch_count):
        sim_name = f"{training_name}-{n:03d}"
        opponent = sr.ControllerProvider.get(opponent_name)

        sim_info = None
        if record:
            sim_info = sr.SimInfo(
                name1=f"{loop_name}-agent",
                desc1={"info": f"Agent with {loop_name} actions"},
                name2=opponent.name(),
                desc2=opponent.description(),
                port=port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )

        env = sgym.SEnv(
            senv_config=sgym.default_senv_config,
            senv_mapping=cont_sgym_mapping,
            port=port,
            sim_name=sim_name,
            opponent=opponent,
            reward_handler=reward_handler,
            sim_info=sim_info,
        )
        _observation, _info = env.reset()
        cnt = 0
        episode_over = False
        cuml_reward = 0.0
        while not episode_over:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            cuml_reward += reward
            episode_over = terminated or truncated
            cnt += 1

        print(
            f"### finished epoch {sim_name} "
            f"reward:{cuml_reward:10.2f} record:{record}"
        )
        rewards.append(cuml_reward)
        env.close()

    out_dir = Path.home() / "tmp" / "sumosim"
    out_dir.mkdir(exist_ok=True, parents=True)

    desc = {
        "loop name": "q-sample",
        "reward handler": reward_handler_name.value,
        "epoch count": epoch_count,
    }
    lines = sh.create_lines(desc, [[0], [1, 2]])
    sh.boxplot(rewards, out_dir, training_name, lines)
    print(f"Wrote plot {training_name} to {out_dir}")


def _tid() -> str:
    return f"{int(time.time() * 10) % 86400:05d}"


def _continuous_to_discrete(
    value: float, min_value: float, max_value: float, step_count: int
) -> int:
    d = (max_value - min_value) / step_count
    i = int(math.floor((value + max_value) / d))
    return min(max(0, i), (step_count - 1))


def _create_subset(max_value: float, n: int) -> list[float]:
    diff = max_value / n
    return [-max_value + i * diff for i in range(2 * n + 1)]


def _create_q_observation_space() -> gym.Space:
    pass


def _to_q_observation(view: int, border: np.array) -> tuple[int, tuple[int, int, int]]:
    pass


def _create_numpy_array(value: any, config: sgym.SEnvConfig) -> np.array:
    return np.array(value, dtype=config.dtype)
