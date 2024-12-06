from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tryout():
    name = "Q-TRAIN-try001-51045"
    in_dir = Path.home() / "tmp" / "sumosim"
    file_path = in_dir / f"{name}.json"
    data = pd.read_json(file_path)
    y = data["reward"]

    window_size = 1000
    y1 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")
    window_size = 10000
    y2 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

    # plt.plot(y)
    plt.plot(y1, label="reward")
    plt.plot(y2, label="reward (flat)")
    plt.title(name)
    plt.legend()
    plt.show()
