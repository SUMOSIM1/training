import math
import time
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np

import training.sgym as sgym
import training.simrunner as sr
import training.sumosim_helper as sh


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

        print(f"### finished epoch {sim_name} cr:{cuml_reward:10.2f} r:{record}")
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

        print(f"### finished epoch {sim_name} r:{cuml_reward:10.2f} r:{record}")
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
