import math
import time
from pathlib import Path

import pandas as pd


def row_col(n: int) -> (int, int):
    if n < 4:
        return 1, n
    rows = int(math.ceil(math.sqrt(float(n))))
    cols = int(math.ceil(float(n) / rows))
    return rows, cols


def write_dict_data(data: list, out_dir: Path, name: str) -> Path:
    filename = f"{name}.json"
    file = out_dir / filename
    df = pd.DataFrame(data)
    df.to_json(file, indent=2)
    return file


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


def time_id() -> str:
    return f"{int(time.time() * 10) % 86400:05d}"


def cont_to_discrete(
    value: float, min_value: float, max_value: float, step_count: int
) -> int:
    d = (max_value - min_value) / step_count
    i = int(math.floor((value - min_value) / d))
    return min(max(0, i), (step_count - 1))


def cont_values(min_value: float, max_value: float, n: int) -> list[float]:
    diff = (max_value - min_value) / (n - 1)
    return [min_value + i * diff for i in range(n)]
