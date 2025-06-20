import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import cast
from pathlib import Path
from enum import Enum

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sbn
import yaml
import pprint as pp

import training.helper as hlp
import training.sgym.core as sgym
import training.simrunner as sr
import training.sgym.sim_mapping as sm
import training.sgym.qtables as qt
import training.reward.reward_core as rhc


class FetchType(str, Enum):
    EAGER = "eager"
    LAZY_S = "lazy-s"
    LAZY_M = "lazy-m"
    LAZY_L = "lazy-l"
    LAZY_S_T2 = "lazy-s-t2"
    LAZY_S_T5 = "lazy-s-t5"
    LAZY_S_T10 = "lazy-s-t10"
    LAZY_M_T2 = "lazy-m-t2"
    LAZY_M_T5 = "lazy-m-t5"
    LAZY_M_T10 = "lazy-m-t10"
    LAZY_L_T2 = "lazy-l-t2"
    LAZY_L_T5 = "lazy-l-t5"
    LAZY_L_T10 = "lazy-l-t10"

    def fetch(self, row: np.array) -> int:
        match self:
            case FetchType.EAGER:
                return int(np.argmax(row))
            case FetchType.LAZY_S:
                return self.lazy(0.0005, row)
            case FetchType.LAZY_M:
                return self.lazy(0.005, row)
            case FetchType.LAZY_L:
                return self.lazy(0.01, row)
            case FetchType.LAZY_S_T2:
                return self.lazy_top(0.0005, 2, row)
            case FetchType.LAZY_S_T5:
                return self.lazy_top(0.0005, 5, row)
            case FetchType.LAZY_S_T10:
                return self.lazy_top(0.0005, 10, row)
            case FetchType.LAZY_M_T2:
                return self.lazy_top(0.005, 2, row)
            case FetchType.LAZY_M_T5:
                return self.lazy_top(0.005, 5, row)
            case FetchType.LAZY_M_T10:
                return self.lazy_top(0.005, 10, row)
            case FetchType.LAZY_L_T2:
                return self.lazy_top(0.01, 2, row)
            case FetchType.LAZY_L_T5:
                return self.lazy_top(0.01, 5, row)
            case FetchType.LAZY_L_T10:
                return self.lazy_top(0.01, 10, row)
            case _:
                raise RuntimeError(
                    f"Action for fetch type {self.value} not yet implemented"
                )

    @staticmethod
    def lazy(eps: float, row: np.array) -> int:
        q_max = np.max(row)
        border = q_max - eps
        filtered = np.where(row > border)
        return np.random.choice(filtered[0])

    @staticmethod
    def lazy_top(eps: float, top: int, row: np.array) -> int:
        q_max = np.max(row)
        border = q_max - eps
        ixs = np.where(row > border)[0]
        tuples = [(ix, row[ix]) for ix in ixs]
        _sorted = sorted(tuples, key=lambda t: -t[1])
        _top = [t[0] for t in _sorted[0:top]]
        return np.random.choice(_top)


class EpsilonDecay(str, Enum):
    NONE = "none"
    DECAY_100_80 = "decay-100-80"
    DECAY_100_50 = "decay-100-50"
    DECAY_100_20 = "decay-100-20"
    DECAY_100_10 = "decay-100-10"
    DECAY_100_05 = "decay-100-05"
    DECAY_1000_80 = "decay-1000-80"
    DECAY_1000_50 = "decay-1000-50"
    DECAY_1000_20 = "decay-1000-20"
    DECAY_1000_10 = "decay-1000-10"
    DECAY_1000_05 = "decay-1000-05"
    DECAY_3000_80 = "decay-3000-80"
    DECAY_3000_50 = "decay-3000-50"
    DECAY_3000_20 = "decay-3000-20"
    DECAY_3000_10 = "decay-3000-10"
    DECAY_3000_05 = "decay-3000-05"
    DECAY_EXP_100 = "decay-exp-100"
    DECAY_EXP_1000 = "decay-exp-1000"
    DECAY_EXP_5000 = "decay-exp-5000"

    def epsilon(self, epoch: int, initial_epsilon: float) -> float:
        match self:
            case EpsilonDecay.NONE:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 0, 1.0)
            case EpsilonDecay.DECAY_100_05:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 100, 0.05)
            case EpsilonDecay.DECAY_100_10:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 100, 0.1)
            case EpsilonDecay.DECAY_100_20:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 100, 0.2)
            case EpsilonDecay.DECAY_100_50:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 100, 0.5)
            case EpsilonDecay.DECAY_100_80:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 100, 0.8)
            case EpsilonDecay.DECAY_1000_05:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 1000, 0.05)
            case EpsilonDecay.DECAY_1000_10:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 1000, 0.1)
            case EpsilonDecay.DECAY_1000_20:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 1000, 0.2)
            case EpsilonDecay.DECAY_1000_50:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 1000, 0.5)
            case EpsilonDecay.DECAY_1000_80:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 1000, 0.8)
            case EpsilonDecay.DECAY_3000_05:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 3000, 0.05)
            case EpsilonDecay.DECAY_3000_10:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 3000, 0.1)
            case EpsilonDecay.DECAY_3000_20:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 3000, 0.2)
            case EpsilonDecay.DECAY_3000_50:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 3000, 0.5)
            case EpsilonDecay.DECAY_3000_80:
                return self._epsilon_decay_linear(epoch, initial_epsilon, 3000, 0.8)
            case EpsilonDecay.DECAY_EXP_100:
                return self._epsilon_decay_exp(epoch, initial_epsilon, 100)
            case EpsilonDecay.DECAY_EXP_1000:
                return self._epsilon_decay_exp(epoch, initial_epsilon, 1000)
            case EpsilonDecay.DECAY_EXP_5000:
                return self._epsilon_decay_exp(epoch, initial_epsilon, 5000)
            case _:
                raise RuntimeError(f"EpsilonDecay for {self.value} not yet implemented")

    @staticmethod
    def _epsilon_decay_linear(
        epoch: int, initial_epsilon: float, decay_epochs: int, decay_factor: float
    ):
        if epoch >= decay_epochs:
            return initial_epsilon * decay_factor
        k = (initial_epsilon * (1.0 - decay_factor)) / decay_epochs
        d = k * epoch
        _re = initial_epsilon - d
        return _re

    @staticmethod
    def _epsilon_decay_exp(epoch: int, initial_epsilon: float, half_time: float):
        return hlp.descending_exponential(
            start_value=initial_epsilon, half_time=half_time, time=float(epoch)
        )


@dataclass(frozen=True)
class QLearnConfig:
    learning_rate: float
    epsilon: float
    epsilon_decay_name: str
    discount_factor: float
    mapping_name: str
    opponent_name: str
    reward_handler_name: str
    fetch_type: str


default_senv_config = sgym.SEnvConfig(
    max_wheel_speed=7,
    wheel_speed_steps=10,
    max_view_distance=700,
    view_distance_steps=3,
    opponent_see_steps=4,
    max_simulation_steps=1000,
    dtype=np.float32,
)


default_q_learn_config = QLearnConfig(
    learning_rate=0.1,
    epsilon=0.1,
    epsilon_decay_name=EpsilonDecay.NONE.value,
    discount_factor=0.8,
    mapping_name=sm.SEnvMappingName.NON_LINEAR_3.value,
    opponent_name=sr.ControllerName.STAND_STILL.value,
    reward_handler_name=rhc.RewardHandlerName.CONTINUOUS_CONSIDER_ALL.value,
    fetch_type=FetchType.EAGER.value,
)


def q_learn(
    name: str,
    epoch_count: int,
    sim_host: str,
    sim_port: int,
    db_host: str,
    db_port: int,
    record: bool,
    out_dir: str,
    q_learn_config: QLearnConfig,
) -> int:
    print(f"--- q_learn {name} {pp.pformat(q_learn_config)}")
    senv_config: sgym.SEnvConfig = default_senv_config
    reward_handler = rhc.RewardHandlerProvider.get(
        rhc.RewardHandlerName(q_learn_config.reward_handler_name)
    )
    fetch_type = FetchType(q_learn_config.fetch_type)
    results = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    loop_name = "q"

    doc_interval = calc_doc_interval(epoch_count)
    doc_duration = calc_doc_duration(doc_interval)
    record_count = calc_record_count(epoch_count)

    record_interval = max(1, epoch_count // record_count)
    document_config(name, epoch_count, record, q_learn_config, out_path)
    opponent = sr.ControllerProvider.get(
        sr.ControllerName(q_learn_config.opponent_name)
    )
    env = sgym.SEnv(
        senv_config=senv_config,
        senv_mapping=sm.senv_mapping(
            sm.SEnvMappingName(q_learn_config.mapping_name), senv_config
        ),
        sim_host=sim_host,
        sim_port=sim_port,
        db_host=db_host,
        db_port=db_port,
        opponent=opponent,
        reward_handler=reward_handler,
    )
    agent = QAgent(
        env=env,
        reward_handler=rhc.RewardHandlerName(q_learn_config.reward_handler_name),
        learning_rate=q_learn_config.learning_rate,
        initial_epsilon=q_learn_config.epsilon,
        epsilon_decay=EpsilonDecay(q_learn_config.epsilon_decay_name),
        discount_factor=q_learn_config.discount_factor,
        fetch_type=fetch_type,
    )
    for epoch_nr in range(epoch_count):
        sim_name = f"{name}-{epoch_nr:06d}"
        sim_info = None
        if record and (
            epoch_nr % record_interval == 0 or is_last(epoch_count, epoch_nr)
        ):
            sim_info = sr.SimInfo(
                name1=f"{loop_name}-agent",
                desc1={"info": f"Agent with {loop_name} actions"},
                name2=opponent.name(),
                desc2=opponent.description(),
                port=sim_port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )
        obs, _info = env.reset(sim_info, sim_name)
        sim_nr = 0
        episode_over = False
        cuml_reward = 0.0
        while not episode_over:
            action = agent.get_action(obs, epoch_nr)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # print(f"# obs:{obs} a:{action} next_obs:{next_obs}")
            agent.update(obs, action, float(reward), terminated, next_obs)
            cuml_reward += reward
            episode_over = terminated or truncated
            obs = next_obs
            sim_nr += 1

        results.append(
            {
                "sim_steps": sim_nr,
                "max_sim_steps": senv_config.max_simulation_steps,
                "epoch_nr": epoch_nr,
                "max_epoch_nr": epoch_count,
                "reward": cuml_reward,
            }
        )
        if epoch_nr % (max(1, doc_interval // 10)) == 0:
            progr = hlp.progress_str(epoch_nr, epoch_count)
            print(f"Finished epoch {name} {progr} reward:{cuml_reward:.5f}")
        if do_plot_q_values(epoch_nr, doc_interval, doc_duration) or is_last(
            epoch_count, epoch_nr
        ):
            document_q_values(name, agent, epoch_nr, sim_nr, out_path, senv_config)
        if (epoch_nr % doc_interval == 0 and epoch_nr > 0) or is_last(
            epoch_count, epoch_nr
        ):
            document(name, results, epoch_nr, q_learn_config, out_path)
    env.close()
    print(f"Finished training {name} {loop_name} p:{sim_port}")
    return sim_port


def calc_doc_interval(epoch_count: int) -> int:
    if epoch_count < 50:
        return 1
    elif epoch_count < 1000:
        return 10
    elif epoch_count < 10000:
        return 100
    else:
        return 1000


def calc_doc_duration(doc_interval: int) -> int:
    if doc_interval <= 100:
        return doc_interval
    elif doc_interval <= 1000:
        return doc_interval // 10
    else:
        return doc_interval // 100


def calc_record_count(epoch_count: int) -> int:
    if epoch_count < 20:
        return epoch_count
    else:
        return 10


def is_last(epoch_count, epoch_nr):
    return epoch_nr == (epoch_count - 1)


def initial_rewards(n: int) -> list[float]:
    return list(
        np.random.rand(
            n,
        )
        * 0.001
    )


def calc_next_q_value(
    reward: float,
    terminated: float,
    next_obs_q_values: list[float],
    current_q_value: float,
    discount_factor: float,
    learning_rate: float,
) -> tuple[float, float]:
    future_q_value = (not terminated) * np.max(next_obs_q_values)
    temporal_difference = reward + discount_factor * future_q_value - current_q_value
    q_value = current_q_value + learning_rate * temporal_difference
    return temporal_difference, q_value


def adjust_end(reward: float) -> float:
    min_value = -150.0
    max_value = 220.0
    _r = (reward - min_value) / (max_value - min_value)
    return min(max(0.0, _r), 1.0)


def adjust_cont(reward: float) -> float:
    min_value = -160.0
    max_value = 660.0
    _r = (reward - min_value) / (max_value - min_value)
    return min(max(0.0, _r), 1.0)


class QAgent:
    def __init__(
        self,
        env: gym.Env,
        reward_handler: rhc.RewardHandlerName,
        fetch_type: FetchType,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: EpsilonDecay,
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
        self.q_values = defaultdict(lambda: initial_rewards(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.reward_handler = reward_handler
        self.fetch_type = fetch_type

        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay

        self.training_error = []

    def get_action(self, obs: tuple, epoch: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon_decay.epsilon(epoch, self.initial_epsilon):
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            q_values_row = self.q_values[obs]
            return self.fetch_type.fetch(q_values_row)

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """Updates the Q-value of an action."""

        match self.reward_handler:
            case rhc.RewardHandlerName.END_CONSIDER_ALL:
                reward = adjust_end(reward)
            case _:
                reward = adjust_cont(reward)

        temporal_difference, self.q_values[obs][action] = calc_next_q_value(
            reward,
            terminated,
            self.q_values[next_obs],
            self.q_values[obs][action],
            self.discount_factor,
            self.lr,
        )
        self.training_error.append(temporal_difference)


def do_plot_q_values(n: int, interval: int, duration: int) -> bool:
    return bool(n % interval < duration)


def document_q_values(
    name: str,
    agent: QAgent,
    epoch_nr: int,
    sim_nr: int,
    work_dir: Path,
    q_learn_env_config: sgym.SEnvConfig,
):
    work_dir = work_dir / "q-value-heat"
    work_dir.mkdir(parents=True, exist_ok=True)
    plot_q_values(agent, epoch_nr, sim_nr, name, work_dir, q_learn_env_config)


def document(
    name: str, results: list[dict], epoch_nr: int, config: QLearnConfig, work_dir: Path
):
    data_path = work_dir / f"{name}.json"
    df = pd.DataFrame(results)
    df.to_json(data_path, indent=2)
    plot_boxplot(df, name, config, work_dir)
    plot_all(df, name, config, work_dir)
    # print(f"Wrote plots for {name} to {work_dir.absolute()}")


def document_config(
    name: str,
    epoch_count: int,
    record: bool,
    q_learn_config: QLearnConfig,
    out_dir: Path,
):
    out_file = out_dir / f" {name}-config.yml"
    conf_dict = {
        "name": name,
        "epoch_count": epoch_count,
        "record": record,
        "config": asdict(cast(dataclass(), q_learn_config)),
    }
    with out_file.open("w") as f:
        yaml.dump(conf_dict, f)
    pass


def plot_q_values(
    agent: QAgent,
    epoch_nr: int,
    sim_nr: int,
    name: str,
    work_dir: Path,
    q_learn_env_config: sgym.SEnvConfig,
    report_q_table: bool = False,
) -> Path:
    matplotlib.use("agg")

    _all = qt.all_obs(q_learn_env_config)
    obs_action_data = []
    for obs in _all:
        values = agent.q_values[obs]
        obs_action_data.append(values)
    obs_action_matrix = np.matrix(obs_action_data, dtype=q_learn_env_config.dtype)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    sbn.heatmap(obs_action_matrix, vmin=-1.0, vmax=2.0, ax=ax)
    ax.set_title(f"Q Values for {name} {epoch_nr:08d}")
    ax.set_xlabel("action")
    ax.set_ylabel("observation")

    out_path = work_dir / f"{name}-{epoch_nr:08d}{sim_nr:08d}-heat.png"
    fig.savefig(out_path)
    plt.close(fig)

    if report_q_table:
        json_dict = qt.to_json(agent.q_values, _all)
        all_data = {"epoch_nr": epoch_nr, "sim_nr": sim_nr, "q_values": json_dict}
        data_out_path = work_dir / f"{name}-{epoch_nr:08d}{sim_nr:08d}-data.json"
        with data_out_path.open("w") as f:
            json.dump(all_data, f)
        print(f"Wrote data to {data_out_path}")

    return out_path


def title(column: str, name: str, config: QLearnConfig) -> str:
    lines = [
        f"{column} {name} ",
    ]
    return "\n".join(lines)


def plot_boxplot(
    data: pd.DataFrame, name: str, config: QLearnConfig, work_dir: Path
) -> Path:
    matplotlib.use("agg")
    column = "reward"
    y = data[column]

    x1, y1, medians = hlp.split_data(y, 15)
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        ax.boxplot(y1, labels=x1)
        if medians is not None:
            ax.plot(
                range(1, len(x1) + 1), medians, color="red", linewidth=4, linestyle=":"
            )
        ax.set_title(title(column, name, config))
        ax.set_ylim(ymin=-300, ymax=300)
        ax.set_ylabel(column)
        ax.set_xlabel("epoch nr")
        ax.tick_params(axis="x", rotation=45)
        out_path = work_dir / f"{name}-boxplot.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except ValueError as ve:
        print(f"x1:{x1}")
        print(f"y1:{y1}")
        raise ve


def plot_all(
    data: pd.DataFrame, name: str, config: QLearnConfig, work_dir: Path
) -> Path:
    matplotlib.use("agg")
    column = "reward"
    y = data[column]
    window_size = 1
    y1 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")
    window_size = 10
    y2 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.plot(y1, label="reward")
    ax.plot(y2, label="reward (flat)")
    ax.set_title(title(column, name, config))
    out_path = work_dir / f"{name}-all.png"

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_plain(
    data: pd.DataFrame, name: str, epoch_nr: int, config: QLearnConfig, work_dir: Path
) -> Path:
    matplotlib.use("agg")
    column = "reward"
    y = data[column]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(y, label="reward")
    ax.set_title(title(column, name, config))
    out_path = work_dir / f"{name}-{epoch_nr:05d}plain.png"

    fig.savefig(out_path)
    plt.close(fig)
    return out_path
