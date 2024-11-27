import time
from pathlib import Path

import pandas as pd

import training.sumosim_helper as sh
from training.sgym_training import _tid


def test_tid():
    for i in range(20):
        t = _tid()
        print(f"### tid {i:3d} {t}")
        time.sleep(0.111111)


def test_display_results():
    base_dir = Path.home() / "tmp" / "sumosim"
    filename = "COMBI-06.json"
    file = base_dir / filename
    result = pd.read_json(file)
    sh.plot_epoch_datas(result, base_dir, filename, "")


def main():
    test_display_results()
