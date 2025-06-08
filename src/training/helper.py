import math
import uuid
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import subprocess as sp
import markdown as md
import easing_functions as easing


_fn4 = easing.QuinticEaseIn()
_fn3 = easing.QuarticEaseIn()
_fn2 = easing.CubicEaseIn()
_fn1 = easing.QuadEaseIn()
_fp1 = easing.QuadEaseOut()
_fp2 = easing.CubicEaseOut()
_fp3 = easing.QuarticEaseOut()
_fp4 = easing.QuinticEaseOut()


def row_col(n: int) -> tuple[int, int]:
    if n < 4:
        return 1, n
    rows = int(math.ceil(math.sqrt(float(n))))
    cols = int(math.ceil(float(n) / rows))
    return rows, cols


def write_dict_data(data: pd.DataFrame, out_dir: Path, name: str) -> Path:
    filename = f"{name}.json"
    file = out_dir / filename
    data.to_json(file, indent=2)
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


def unique() -> str:
    return str(uuid.uuid4())[0:6]


def cont_to_discrete(
    x: float, min_value: float, max_value: float, step_count: int, linear: int
) -> int:
    def func(_x: float) -> any:
        if linear <= -4:
            return _fn4(_x)
        if linear == -3:
            return _fn3(_x)
        if linear == -2:
            return _fn2(_x)
        if linear == -1:
            return _fn1(_x)
        if linear == 0:
            return _x
        if linear == 1:
            return _fp1(_x)
        if linear == 2:
            return _fp2(_x)
        if linear == 3:
            return _fp3(_x)
        if linear >= 4:
            return _fp4(_x)

    x1 = (x - min_value) / (max_value - min_value)
    y1 = float(func(x1))
    d = 1.0 / step_count
    i = int(math.floor((y1 / d)))
    return min(max(0, i), (step_count - 1))


def cont_values(min_value: float, max_value: float, n: int) -> list[float]:
    diff = (max_value - min_value) / (n - 1)
    return [min_value + i * diff for i in range(n)]


def compress_means(data: list[float], n: int) -> tuple[list[float], list[float]]:
    data_len = len(data)
    if data_len <= n:
        return list(range(data_len)), data
    d = np.array(data)
    cropped = (data_len // n) * n
    split = np.split(d[0:cropped], n)
    diff = cropped // n
    xs = list(range(0, cropped, diff))
    return xs, np.mean(split, axis=1)


def progress_str(nr: int, count: int) -> str:
    return f"{nr + 1}/{count}"


def create_values(n: int, min: float, max: float) -> list[float]:
    diff = (max - min) / (n - 1)
    return [min + x * diff for x in range(n)]


def parse_integers(integers: str) -> list[int]:
    if not integers:
        return []
    split = integers.split(",")
    return [int(i.strip()) for i in split]


def call(command: list[str], ignore_stderr: bool = False) -> str:
    process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    b_out, b_err = process.communicate()
    if b_err and not ignore_stderr:
        cmd_str = " ".join(command)
        msg = f"ERROR: calling '{cmd_str}'. \n{b_err.decode()}"
        raise RuntimeError(msg)
    if ignore_stderr:
        return b_out.decode() + b_err.decode()
    return b_out.decode()


def call1(command: list[str], work_path: Path | None = None) -> tuple[bool, str]:
    if work_path is not None:
        if not work_path.exists():
            raise RuntimeError(
                f"Cannot call {' '.join(command)}\nbecause workdir {work_path} does not exist"
            )
        process = sp.Popen(
            command, stdout=sp.PIPE, stderr=sp.PIPE, cwd=work_path.absolute()
        )
    else:
        process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    b_out, b_err = process.communicate()
    rc = process.returncode == 0
    return rc, b_out.decode() + b_err.decode()


def parse_markdown(md_txt: str) -> str:
    return md.markdown(md_txt, extensions=["tables"])


def split_data(
    _data: list[float], n: int
) -> tuple[list[str], list[list[float]], list[float] | None]:
    data_len = len(_data)
    if data_len < 10 * n:
        # Less than 10 data per boxplot
        return [str(data_len)], [_data], None
    np_data = np.array(_data)
    cropped = (data_len // n) * n
    split = cast(list[list[float]], np.split(np_data[0:cropped], n))
    medians = [float(np.median(d)) for d in split]
    diff = cropped // n
    xs = range(0, cropped, diff)
    xs_str = [str(x) for x in xs]
    return xs_str, split, medians


def descending_exponential(start_value, half_time, t):
    if half_time <= 0:
        raise ValueError("Half-time must be a positive number.")

    # The formula for a descending exponential function with a half-time is:
    # f(t) = start_value * (1/2)^(t / half_time)
    return start_value * (0.5) ** (t / half_time)
