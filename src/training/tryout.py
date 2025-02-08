from pathlib import Path
import math
import numpy as np

import matplotlib.pyplot as plt
import training.helper as hlp


def func2(x: float, g: float) -> float:
    f = math.pow(10, g)
    return math.pow(f, x - 1) + (math.pow(f, - 1) * (x - 1))


def cont_to_discrete(
    x: float, min_value: float, max_value: float, step_count: int, linear: float
) -> int:
    x1 = (x - min_value) / (max_value - min_value) 
    y1 = func2(x1, linear)
    d = 1.0 / step_count
    i = int(math.floor((y1 / d)))
    return min(max(0, i), (step_count - 1))


def nonlin():
    work_dir = Path.home() / "tmp"

    def func(x: float, g: float) -> float:
        f = math.pow(10, g)
        return math.pow(f, x - 1) + (math.pow(f, - 1) * (x - 1))

    dist = 100
    n = 4
    gs = [0, 0.5, 1, 1.5, 2]
    x = list([a / float(dist - 1) for a in range(dist)])

    x1 = [a / (n - 1) for a in range(n)]
    for g in gs:
        y1 = [func(x, g) for x in x1]
        y1_str = ", ".join([f"{y:5.3f}" for y in y1])
        print(f"{g:6.1f} {y1_str}")

    fig, ax = plt.subplots(figsize=(15, 15))
    for g in gs:
        y1 = [func(a, g) for a in x]
        ax.plot(x, y1)
    out_path = work_dir / "a.png"
    fig.savefig(out_path)
    plt.close(fig)




def main():
    work_dir = Path.home() / "tmp"
    min_value = 0.0
    max_value = 800.0
    dist = 300
    n = 5
    xs = np.arange(min_value, max_value, (max_value - min_value) / (dist - 0.999999)   )
    fig, ax = plt.subplots(figsize=(15, 15))
    ls = [0, 1, 2, 3, 4]
    for l in ls:
        ys = [cont_to_discrete(x, min_value, max_value, n, l) for x in xs]
        ax.plot(xs, ys)
    out_path = work_dir / "b.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote step plot to '{out_path}'")
