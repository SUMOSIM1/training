import typer
from typing_extensions import Annotated
from pathlib import Path
import simrunner as sr

app = typer.Typer()

simpath_help = "Path to the simulator git module"
simpath_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"

@app.command()
def sim(
        simpath: Annotated[Path, typer.Argument(help=simpath_help)] = simpath_default,
        ):
    try:
        sr.run(simpath)
    except Exception as e:
        print(f"ERROR: {e}")

@app.command()
def tryout():
    print(f"tryout")



if __name__ == "__main__":
    app()
