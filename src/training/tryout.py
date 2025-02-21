from pathlib import Path
import pandas as pd
import training.sgym.qlearn as ql


def main():
    result_dirs = [
        "results-2025-01-000",
        "results-2025-01-002",
        "results-2025-01-003",
        "results-2025-02-Q-CV5",
        "results-2025-02-Q-CV7",
        "results-2025-02-QMAP01",
        "results-2025-02-QMAP02",
        "results-2025-02-QMAP03",
    ]

    config = ql.default_q_learn_config
    base_dir = Path.home() / "tmp" / "sumosim" / "results"
    for rdir in result_dirs:
        work_dir = base_dir / rdir
        print(f"--- work_dir {work_dir}")
        for file in work_dir.iterdir():
            if file.name.endswith("json"):
                print(f"creating boxplots for {file}")
                df = pd.read_json(file)
                ql.plot_boxplot(df, file.stem, config, work_dir)
