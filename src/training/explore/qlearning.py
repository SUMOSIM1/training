import time
from datetime import datetime
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path

import training.explore.blackjack as bj
import training.helper as hlp
import training.sgym.qlearn as sgym_qlearn


def tryout01():
    s = datetime.now()
    count = 20
    for i in range(count):
        time.sleep(0.287362)
        print(f"{hlp.progress_str(i, count, s)}")

    print(f"finished at:{datetime.now()}")


def tryout03():
    bj.main()


def tryout02():
    n = 121
    x = sgym_qlearn.initial_rewards(n)
    for i, v in enumerate(x):
        print(f"-- {i} {v}")


def tryout():
    def next(reward: float, current_q: float, next_obs_q: float) -> float:
        _, next_q = sgym_qlearn.calc_next_q_value(
            reward=reward,
            terminated=False,
            next_obs_q_values=[next_obs_q],
            current_q_value=current_q,
            discount_factor=0.95,
            learning_rate=0.01,
        )
        return next_q

    def data(next_obs_q: float):
        def rewards(n: int) -> list[float]:
            min = -170
            max = 170
            diff = (max - min) / (n - 1)
            return [min + x * diff for x in range(n)]

        def values(n: int) -> list[float]:
            min = -5.0
            max = 5.0
            diff = (max - min) / (n - 1)
            return [min + x * diff for x in range(n)]

        _rewards = rewards(5)
        _curr_values = values(5)
        result = []
        for cv in _curr_values:
            for r in _rewards:
                next_q = next(reward=r, current_q=cv, next_obs_q=next_obs_q)
                result.append({"curr_q": cv, "next_q": next_q, "reward": r})
        data = pd.DataFrame(result)
        return data.pivot(index="reward", columns="curr_q", values="next_q")

    nrows = 3
    ncols = 3
    next_obs_qs = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(20, 20), constrained_layout=True
    )
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if k < len(next_obs_qs):
                ax = axs[i][j]
                nq = next_obs_qs[k]
                _data = data(next_obs_q=nq)
                ax = sb.lineplot(_data, ax=ax)
                ax.set_title(f"next obs q {nq:.3f}")
                ax.set_ylabel("next_q")
                ax.set_xlim(-200, 200)
                ax.set_ylim(-15, 15)
                ax.grid()
                k += 1
    dir = Path.home() / "tmp" / "sumosim" / "tryout"
    dir.mkdir(exist_ok=True, parents=True)
    file = dir / "next_q_value.png"
    fig.savefig(file)
    print(f"Wrote analysis to {file}")
