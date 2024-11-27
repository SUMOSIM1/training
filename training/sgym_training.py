import math
import time
from enum import Enum

import training.sgym as sgym
import training.simrunner as sr


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

    for n in range(epoch_count):
        sim_name = f"SGYM-001-{run_id}-{n:03d}"
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
        observation, info = env.reset()
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
        env.close()


def q_sample(
    epoch_count: int,
    record: bool,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    raise NotImplementedError()


def _tid() -> str:
    return f"{int(time.time() * 10) % 86400:05d}"


def _continuous_to_discrete(value: float, max_value: float, step_count: int) -> int:
    d = 2.0 * max_value / step_count
    i = int(math.floor((value + max_value) / d))
    return min(max(0, i), (step_count - 1))
