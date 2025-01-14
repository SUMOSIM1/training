from pathlib import Path
import yaml
from pprint import pprint


def sumosim_report_path(dir: str) -> Path:
    p = Path(dir)
    if not p.is_absolute():
        p = Path.home() / p
    if not p.exists():
        raise ValueError(f"path {dir} does not exit")
    return p


def results(result_dir: Path, name_prefix: str, result_name: str) -> list[Path]:
    results = [
        r.absolute()
        for r in result_dir.iterdir()
        if r.name.startswith(name_prefix) and r.stem.endswith(result_name)
    ]
    return sorted(results)


def report(name_prefix: str, results_dir: str):
    print(f"Creating report for {name_prefix}")
    report_path = (
        Path(__file__).parent.parent.parent.parent / "resources" / "report.yml"
    )
    with report_path.open() as f:
        reports_data = yaml.safe_load(f)
    filtered_reports = [r for r in reports_data if r["prefix"].startswith(name_prefix)]
    pprint(filtered_reports)
    vp = sumosim_report_path(results_dir)
    videos = results(vp, name_prefix, "sumosim-video")
    for v in videos:
        print(v.name)
    q_values = results(vp, name_prefix, "q-values-heat")
    for r in q_values:
        print(r.name)
    boxplots = results(vp, name_prefix, "boxplot")
    for r in boxplots:
        print(r.name)
