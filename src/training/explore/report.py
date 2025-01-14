from pathlib import Path
import yaml
from pprint import pprint

def report(name_prefix: str):
    print(f"Creating report for {name_prefix}")
    report_path = Path(__file__).parent.parent.parent.parent / "resources" / "report.yml"
    with report_path.open() as f:
        reports = yaml.safe_load(f)
        filtered_reports = [r for r in reports if r['prefix'].startswith(name_prefix) ]
        pprint(filtered_reports)