from pathlib import Path
import subprocess as sp

def run(path: Path):
    if not path.exists():
        raise RuntimeError(f"simpath {path} does not exist.")
    sbtfile = path / "build.sbt"
    if not sbtfile.exists():
        raise RuntimeError(f"simpath {path} contains no 'build.sbt' file.")
    print(f"starting sim in {path}")
    sp.call(['sbt', '--supershell=false', 'sumosimJVM/run udp'], cwd=f"{path}")
