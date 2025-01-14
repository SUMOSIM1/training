from pathlib import Path
import yaml
from pprint import pprint


def _report_path(dir: str) -> Path:
    p = Path(dir)
    if not p.is_absolute():
        p = Path.home() / p
    if not p.exists():
        raise ValueError(f"path {dir} does not exit")
    return p


def report(name_prefix: str, video_dir: str):
    print(f"Creating report for {name_prefix}")
    report_path = Path(__file__).parent.parent.parent.parent / "resources" / "report.yml"
    with report_path.open() as f:
        reports = yaml.safe_load(f)
        filtered_reports = [r for r in reports if r['prefix'].startswith(name_prefix) ]
        pprint(filtered_reports)
    vp = _report_path(video_dir)
    pprint(vp)
