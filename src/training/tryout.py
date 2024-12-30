import pandas as pd
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from pprint import pprint
from dataclasses import dataclass


def reward_values():
    work_dir = Path.home() / "tmp" / "sumosim" / "start"
    file_c = work_dir / "MM-C-20977.json"
    file_e = work_dir / "MM-E-52085.json"
    df1 = pd.read_json(file_c)
    df2 = pd.read_json(file_e)

    def concat_rewards(df: pd.DataFrame) -> list[float]:
        r1 = df["r1"]
        r2 = df["r2"]
        a1 = list([float(x) for x in r1])
        a2 = list([float(x) for x in r2])
        return list(a1) + list(a2)

    data = {"continuous": concat_rewards(df1), "end": concat_rewards(df2)}

    fig, ax = plt.subplots(figsize=(5, 10))
    sb.boxplot(data, ax=ax)
    ax.grid(which="both", color="black", linestyle="dotted", linewidth=0.1)
    ax.yaxis.set_ticks(np.arange(-200, 700, 20))
    ax.set_title("Reward Values Distribution")
    ax.set_xlabel("reward handler")
    out_path = work_dir / "mm.png"
    fig.savefig(out_path)
    print(f"Wrote plot to {out_path}")


@dataclass
class Product:
    name: str
    values: dict


def cv_dict():
    def products(values_dict: dict, max_parallel: int) -> list[list[Product]]:
        def to_dict(keys: list, batch_values: list) -> dict:
            return [dict(zip(keys, value)) for value in batch_values]

        def to_ids(dicts: list[dict]) -> list[str]:
            def to_id(dictionary: dict) -> str:
                return "".join([f"{key}{index}" for key, index in dictionary.items()])

            return [to_id(dictionary) for dictionary in dicts]

        def create_batched_dicts(
            keys: list, lists: list, max_parallel: 20
        ) -> list[list[dict]]:
            prod_values = list(it.product(*lists))
            batch_size = (len(prod_values) + max_parallel - 1) // max_parallel
            batched_values = it.batched(prod_values, batch_size)
            return [to_dict(keys, values) for values in batched_values]

        _keys = values_dict.keys()
        _values = [values_dict[k] for k in _keys]
        batched_dicts = create_batched_dicts(_keys, _values, max_parallel)

        index_values = [range(len(list(value))) for value in _values]
        batched_index_dicts = create_batched_dicts(_keys, index_values, max_parallel)
        ids = [to_ids(d) for d in batched_index_dicts]

        double_zipped = [zip(a, b) for a, b in (zip(batched_dicts, ids))]
        return [
            [Product(values=values, name=name) for values, name in zipped]
            for zipped in double_zipped
        ]

    values_dict = {
        "L": [0.1, 0.01, 0.001],
        "E": [0.1, 0.05, 0.01],
        "D": [0.99, 0.95, 0.5],
    }
    max_parallel = 10
    _products = products(values_dict, max_parallel)
    pprint(_products)

    cut_index = len(_products) // 6 * 5
    range_work = list(range(cut_index))
    range_ben = list(range(cut_index, len(_products)))
    print("range work", range_work)
    print("range ben", range_ben)


def main():
    # cv_values()
    cv_dict()
