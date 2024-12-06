import math
import time
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import gymnasium as gym
import gymnasium.spaces as gyms
import numpy as np

import training.sgym as sgym
import training.simrunner as sr
import training.sumosim_helper as sh
from training.simrunner import DiffDriveValues


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


class SGymLoop(Enum):
    SAMPLE = "sample"
    Q_TRAIN = "q-train"


def main(
    name: str,
    sgym_loop: SGymLoop,
    epoch_count: int,
    record: bool,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    match sgym_loop:
        case SGymLoop.SAMPLE:
            sample(name, epoch_count, record, port, opponent_name, reward_handler_name)
        case SGymLoop.Q_TRAIN:
            q_train(name, epoch_count, record, port, opponent_name, reward_handler_name)


def sample(
    name: str,
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
                port=port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )

        env = sgym.SEnv(
            senv_config=sgym.default_senv_config,
            senv_mapping=cont_sgym_mapping(),
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


def q_train(
    name: str,
    epoch_count: int,
    record: bool,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    loop_name = "q-train"
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    print(
        f"### sgym {loop_name} e:{epoch_count} p:{port} "
        f"o:{opponent_name.value} rh:{reward_handler_name.value} r:{record}"
    )

    run_id = _tid()

    results = []
    training_name = f"Q-TRAIN-{name}-{run_id}"
    for epoch_nr in range(epoch_count):
        sim_name = f"{training_name}-{epoch_nr:06d}"
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

        cfg = sgym.default_senv_config

        env = sgym.SEnv(
            senv_config=cfg,
            senv_mapping=q_sgym_mapping(cfg),
            port=port,
            sim_name=sim_name,
            opponent=opponent,
            reward_handler=reward_handler,
            sim_info=sim_info,
        )

        agent = QAgent(
            env=env,
            learning_rate=0.01,
            initial_epsilon=0.01,
            epsilon_decay=0.001,
            final_epsilon=0.05,
            discount_factor=0.95,
        )

        obs, _info = env.reset()
        cnt = 0
        episode_over = False
        cuml_reward = 0.0
        while not episode_over:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # print(f"# obs:{obs} a:{action} next_obs:{next_obs}")
            agent.update(obs, action, reward, terminated, next_obs)
            cuml_reward += reward
            episode_over = terminated or truncated
            obs = next_obs
            cnt += 1

        print(
            f"### finished epoch {sim_name} "
            f"reward:{cuml_reward:10.2f} record:{record}"
        )
        results.append(
            {
                "sim_steps": cnt,
                "max_sim_steps": cfg.max_simulation_steps,
                "epoch_nr": epoch_nr,
                "max_epoch_nr": epoch_count,
                "reward": cuml_reward,
            }
        )
        env.close()

    out_dir = Path.home() / "tmp" / "sumosim"
    out_dir.mkdir(exist_ok=True, parents=True)

    sh.write_dict_data(results, out_dir, training_name)
    print(f"Wrote data for {training_name} to {out_dir}")


def get_q_act_space(config: sgym.SEnvConfig) -> gym.Space:
    n = config.wheel_speed_steps * config.wheel_speed_steps
    return gyms.Discrete(n)


def get_q_obs_space(config: sgym.SEnvConfig) -> gym.Space:
    return gyms.Tuple(
        (
            gyms.Discrete(n=4),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
        )
    )


def map_q_sensor_to_obs(
    sensor: sr.CombiSensor, config: sgym.SEnvConfig
) -> tuple[int, int, int, int]:
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

    def discrete(distance: float) -> int:
        return _continuous_to_discrete(
            distance, 0.0, config.max_view_distance, config.view_distance_steps
        )

    return (
        view_mapping(),
        discrete(sensor.left_distance),
        discrete(sensor.front_distance),
        discrete(sensor.right_distance),
    )


def curry_q_act_to_diff_drive(
    config: sgym.SEnvConfig,
) -> Callable[[any, sgym.SEnvConfig], sr.DiffDriveValues]:
    velo_from_index = _curry_velo_from_index(
        config.max_wheel_speed, config.wheel_speed_steps
    )

    def inner(a_space: int, _config: sgym.SEnvConfig) -> sr.DiffDriveValues:
        return velo_from_index(a_space)

    return inner


def q_sgym_mapping(cfg: sgym.SEnvConfig) -> sgym.SEnvMapping:
    return sgym.SEnvMapping(
        act_space=get_q_act_space,
        obs_space=get_q_obs_space,
        map_act=curry_q_act_to_diff_drive(cfg),
        map_sensor=map_q_sensor_to_obs,
    )


class QAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.array(self.start_samples()))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def start_samples(self):
        return [self.env.action_space.sample() for _i in range(self.env.action_space.n)]

    def get_action(self, obs: tuple) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def _tid() -> str:
    return f"{int(time.time() * 10) % 86400:05d}"


def _continuous_to_discrete(
    value: float, min_value: float, max_value: float, step_count: int
) -> int:
    d = (max_value - min_value) / step_count
    i = int(math.floor((value - min_value) / d))
    return min(max(0, i), (step_count - 1))


def _cont_values(min_value: float, max_value: float, n: int) -> list[float]:
    diff = (max_value - min_value) / (n - 1)
    return [min_value + i * diff for i in range(n)]


def _create_q_observation_space() -> gym.Space:
    pass


def _to_q_observation(view: int, border: np.array) -> tuple[int, tuple[int, int, int]]:
    pass


def _create_numpy_array(value: any, config: sgym.SEnvConfig) -> np.array:
    return np.array(value, dtype=config.dtype)


def _curry_velo_from_index(
    max_velo: float, velo_steps: int
) -> Callable[[int], sr.DiffDriveValues]:
    velos = _cont_values(-max_velo, max_velo, velo_steps + 1)
    n = len(velos)
    diff_drives = []
    for i in range(n):
        for j in range(n):
            diff_drives.append(DiffDriveValues(velos[i], velos[j]))

    def inner(index: int) -> sr.DiffDriveValues:
        return diff_drives[index]

    return inner
