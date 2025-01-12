from pathlib import Path

def report(name_prefix: str):
    print(f"Creating report for {name_prefix}")
    yml_path = Path(__file__).parent.parent.parent.parent / "resources" / "report.yml"
    print(yml_path, yml_path.exists())