from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import training.helper as hlp


def steps():
    work_dir = Path.home() / "tmp"
    min_value = 0.0
    max_value = 800.0
    dist = 300
    n = 3
    xs = np.arange(min_value, max_value, (max_value - min_value) / (dist - 0.999999))
    fig, ax = plt.subplots(figsize=(15, 15))
    ls = [-4, -3, -2, 1, 2, 3, 4]
    for linear in ls:
        ys = [hlp.cont_to_discrete(x, min_value, max_value, n, linear) for x in xs]
        ax.plot(xs, ys)
    out_path = work_dir / "b.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote step plot to '{out_path}'")


def main():
    steps()
