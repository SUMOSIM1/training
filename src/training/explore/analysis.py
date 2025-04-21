from enum import Enum
from pathlib import Path
from typing import Callable

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import seaborn as sbn
import numpy as np

import training.reward.reward as rw
import training.reward.reward_helper as rh
import training.simrunner_core as src
import training.sgym.qlearn as ql
import training.helper as hlp


class AnalysisName(Enum):
    NEXT_Q_VALUE = "next-q-value"
    ADJUST_REWARD = "adjust-reward"
    VISUALIZE_REWARD = "visualize-reward"
    VISUALIZE_CAN_SEE = "visualize-can-see"


def analysis_main(analysis_name: AnalysisName, out_dir: Path):
    match analysis_name:
        case AnalysisName.NEXT_Q_VALUE:
            next_q_value(out_dir)
        case AnalysisName.ADJUST_REWARD:
            adjust_reward(out_dir)
        case AnalysisName.VISUALIZE_REWARD:
            visualize_reward(out_dir)
        case AnalysisName.VISUALIZE_CAN_SEE:
            visualize_can_see(out_dir)
        case _:
            raise ValueError(f"Unknown analysis name {analysis_name}")


def adjust_reward(out_dir: Path):
    from training.sgym.qlearn import adjust_end
    from training.sgym.qlearn import adjust_cont

    rewards = hlp.create_values(100, -200, 800)
    end_rewards = [adjust_end(x) for x in rewards]
    continuous_rewards = [adjust_cont(x) for x in rewards]

    fig, ax = plt.subplots(figsize=(20, 20))
    sb.lineplot(x=rewards, y=rewards, ax=ax)
    sb.lineplot(x=rewards, y=end_rewards, ax=ax)
    sb.lineplot(x=rewards, y=continuous_rewards, ax=ax)
    ax.set_ylim(-0.1, 1.1)

    work_dir = out_dir / "adjust-reward"
    work_dir.mkdir(parents=True, exist_ok=True)
    file = work_dir / "adjust-reward-001.png"
    fig.savefig(file)
    print(f"Wrote analysis 'adjust-reward to {file}'")


def next_q_value(out_dir: Path):
    n_rows = 3
    n_cols = 3
    work_dir = out_dir / "next-q-value"
    discount_factor = 0.95
    learning_rate = 0.1

    def _next(reward: float, current_q: float, next_obs_q: float) -> float:
        _, _next_q = ql.calc_next_q_value(
            reward=reward,
            terminated=False,
            next_obs_q_values=[next_obs_q],
            current_q_value=current_q,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
        )
        return _next_q

    def plot_next_obs_q(axs: list[list[plt.Axes]]):
        def data(next_obs_q: float):
            _rewards = hlp.create_values(100, 0.0, 1.0)
            _curr_values = hlp.create_values(5, 0.0, 1.0)
            result = []
            for cv in _curr_values:
                for r in _rewards:
                    next_q = _next(reward=r, current_q=cv, next_obs_q=next_obs_q)
                    result.append({"curr_q": cv, "next_q": next_q, "reward": r})
            data = pd.DataFrame(result)
            return data.pivot(index="reward", columns="curr_q", values="next_q")

        next_obs_qs = hlp.create_values(9, 0.0, 0.0)
        k = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if k < len(next_obs_qs):
                    ax = axs[i][j]
                    nq = next_obs_qs[k]
                    _data = data(next_obs_q=nq)
                    ax = sb.lineplot(_data, ax=ax)
                    ax.set_title(f"next obs q {nq:.3f}")
                    ax.set_ylabel("next_q")
                    ax.set_xlim(0.0, 1.0)
                    ax.set_ylim(0.0, 1.0)
                    ax.grid()
                    k += 1

    work_dir.mkdir(exist_ok=True, parents=True)

    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(20, 20), constrained_layout=True
    )
    plot_next_obs_q(axs)
    file = work_dir / "next_q_value.png"
    fig.savefig(file)
    print(f"Wrote next q value analysis to {file}")


def visualize_reward(out_dir: Path):
    reward_handler = rw.CanSeeRewardHandler()

    def _calc_reward(robot1: src.PosDir, robot2: src.PosDir) -> float:
        state = src.SimulationState(robot1, robot2)
        return reward_handler.calculate_reward(state)[0]

    _visualize_see(
        out_dir / "continuous-reward", _calc_reward, (-500, 500), "Continuous Reward"
    )


def visualize_can_see(out_dir: Path):
    def _calc_can_see(robot1: src.PosDir, robot2: src.PosDir) -> float:
        distance = rh.can_see(robot1, robot2)
        return 0.0 if distance is None else distance

    _visualize_see(out_dir / "can-see", _calc_can_see, (-500, 500), "Can See")


def _visualize_see(
    out_dir: Path,
    f: Callable[[src.PosDir, src.PosDir], float],
    see_range: tuple[float, float],
    title: str,
):
    def tick_labels(values: list[float]) -> list[str]:
        def fmt(index: int, num: float) -> str:
            return f"{num:.0f}" if index % 50 == 0 else ""

        return [fmt(i, v) for i, v in enumerate(values)]

    def save_heat_map(index: int, angle: float, out_dir: Path):
        robot1 = src.PosDir(0, 0, angle * math.pi / 180.0)
        ys = np.linspace(see_range[0], see_range[1], num=200)
        matrix = []
        for y in ys:
            xs = np.linspace(see_range[0], see_range[1], num=200)
            row = []
            for x in xs:
                robot2 = src.PosDir(x, y, 0)
                result = f(robot1, robot2)
                (row.append(result),)
            matrix.append(row)
        # pp.pprint(m)
        fig, ax = plt.subplots(figsize=(12, 12))
        sbn.heatmap(
            matrix,
            xticklabels=(tick_labels(xs)),
            yticklabels=(tick_labels(ys)),
            square=True,
            ax=ax,
        )
        ax.set_title(f"{title} - angle: {angle:.2f}")
        ax.set_xlabel(f"x {see_range[0]}, {see_range[0]}")
        ax.set_ylabel(f"y {see_range[0]}, {see_range[0]}")

        out_path = out_dir / f"a-{i:05d}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote to {out_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    angles = [0, 45, 90, 135, 170, 180, 190, 225, 260, 315]
    for i, a in enumerate(angles):
        save_heat_map(i, a, out_dir)
