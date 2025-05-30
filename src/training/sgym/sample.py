from pathlib import Path

import gymnasium as gym
import gymnasium.spaces as gyms
import matplotlib.pyplot as plt
import numpy as np

import training.helper as helper
import training.sgym.core as sgym
import training.simrunner as sr
import training.reward.reward_core as rhc


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
    value = [
        [
            sensor.left_distance,
            sensor.front_distance,
            sensor.right_distance,
        ]
    ]
    return {
        "view": sr.sector_mapping(sensor.opponent_in_sector),
        "border": np.array(value, dtype=config.dtype),
    }


def map_cont_act_to_diff_drive(
    action_space: list[list], config: sgym.SEnvConfig
) -> sr.DiffDriveValues:
    return sr.DiffDriveValues(
        left_velo=action_space[0][1],
        right_velo=action_space[0][0],
    )


def cont_sgym_mapping() -> sgym.SEnvMapping:
    return sgym.SEnvMapping(
        act_space=cont_act_space,
        obs_space=cont_obs_space,
        map_act=map_cont_act_to_diff_drive,
        map_sensor=map_cont_sensor_to_obs,
    )


def sample(
    name: str,
    epoch_count: int,
    record: bool,
    sim_host: str,
    sim_port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: rhc.RewardHandlerName,
):
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    print(
        f"Started sgym sample e:{epoch_count} p:{sim_port} "
        f"o:{opponent_name.value} rh:{reward_handler_name.value} r:{record}"
    )

    run_id = helper.time_id()

    rewards = []
    training_name = f"SGYM-{name}-{run_id}"
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
                port=sim_port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )

        env = sgym.SEnv(
            senv_config=sgym.default_senv_config,
            senv_mapping=cont_sgym_mapping(),
            sim_host=sim_host,
            sim_port=sim_port,
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

        print(f"finished epoch {sim_name} reward:{cuml_reward:10.2f} record:{record}")
        rewards.append(cuml_reward)
        env.close()

    out_dir = Path.home() / "tmp" / "sumosim"
    out_dir.mkdir(exist_ok=True, parents=True)

    desc = {
        "loop name": "sample",
        "reward handler": reward_handler_name.value,
        "epoch count": epoch_count,
    }
    lines = helper.create_lines(desc, [[0], [1, 2]])
    boxplot(rewards, out_dir, training_name, lines)
    print(f"Wrote plot {training_name} to {out_dir}")


def boxplot(data: list[float], out_dir: Path, name: str, suptitle: str) -> Path:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 15))
    title = name
    ax.boxplot(data)
    ax.set_title(title)
    ax.set_ylim([-300, 300])
    plt.suptitle(suptitle, y=0.98)
    filename = f"{name}.png"
    file_path = out_dir / filename
    fig.savefig(file_path)
    plt.close(fig)
    return file_path
