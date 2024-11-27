import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class RewardCollector:
    def __init__(self):
        self.data = []

    def add(self, name1: str, name2: str, value1: float, value2: float):
        self.data.append((name1, name2, value1, value2))

    def data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=["c1", "c2", "r1", "r2"])


def row_col(n: int) -> (int, int):
    if n < 4:
        return 1, n
    rows = int(math.ceil(math.sqrt(float(n))))
    cols = int(math.ceil(float(n) / rows))
    return rows, cols


def create_lines(desc: dict, line_index_list: list[list[int]]) -> str:
    """
    Creates a multiline string from key value pairs
    :param desc: The key value pairs stored in a dict
    :param line_index_list: List of index for lines
    :return:
    """
    k = dict([x for x in enumerate(desc)])

    def elem(i: int) -> str:
        key = k[i]
        value = desc[key]
        return f"{key}:{value}"

    def line(index: list[int]) -> str:
        return " ".join([elem(i) for i in index])

    return "\n".join([line(a) for a in line_index_list])


def write_data(data: RewardCollector, out_dir: Path, name: str):
    filename = f"{name}.json"
    file = out_dir / filename
    data.data_frame().to_json(file, indent=2)


def plot_epoch_datas(
    data: RewardCollector, out_dir: Path, name: str, suptitle: str
) -> Path:
    """
    :param data: A RewardCollector
    :param out_dir:
    :param name:
    :param suptitle:
    :return:
    """
    groups = data.data_frame().groupby(["c1", "c2"])
    num_groups = groups.ngroups

    n_rows, n_cols = row_col(num_groups)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 15))

    group_list = list(groups)
    for i in range(n_cols):
        for j in range(n_rows):
            index = j * n_cols + i
            if n_cols == 1 and n_rows == 1:
                ax = axes
            elif n_cols == 1:
                ax = axes[j]
            elif n_rows == 1:
                ax = axes[i]
            else:
                ax = axes[j][i]
            if index < len(group_list):
                key, grouped_data = group_list[j * n_cols + i]
                name1, name2 = key
                title = f"{name1}  -  {name2}"
                ax.boxplot(grouped_data[["r1", "r2"]], labels=(name1, name2))
                ax.set_title(title)
                ax.set_ylim([-300, 300])
            else:
                ax.axis("off")
    plt.suptitle(suptitle, y=0.98)
    filename = f"{name}.png"
    file_path = out_dir / filename
    fig.savefig(file_path)
    plt.close(fig)
    return file_path


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
