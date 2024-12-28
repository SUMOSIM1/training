from pathlib import Path

import typer
from typing_extensions import Annotated

import training.sgym.qlearn as sgym_qlearn
import training.sgym.sample as sgym_sample
import training.simdb
import training.simrunner as sr
import training.simrunner_tournament as srt
import training.util
import training.tryout as to
import training.explore.analysis as an
import training.explore.export as exp
import training.parallel as prl

app = typer.Typer(pretty_exceptions_enable=False, add_completion=False)

sim_path_help = "Path to the simulator git module"
sim_path_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"


@app.command(help="Runs simulations for combinations of controllers")
def start(
    sim_name: Annotated[str, typer.Option("--name", "-n", help="Simulation name")],
    host: Annotated[
        str,
        typer.Option(
            "--host", "-h", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port", "-p", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    controllers: Annotated[
        list[sr.ControllerName],
        typer.Option("--controllers", "-c", help="Name of controllers"),
    ] = (sr.ControllerName.TUMBLR, sr.ControllerName.BLIND_TUMBLR),
    reward_handler: Annotated[
        sr.RewardHandlerName,
        typer.Option("--reward-handler", help="Name of the reward handler"),
    ] = sr.RewardHandlerName.CONTINUOUS_CONSIDER_ALL,
    combination_type: Annotated[
        srt.CombinationType,
        typer.Option(
            "--combination-type", help="Combination of robots for the tournament"
        ),
    ] = srt.CombinationType.WITHOUT_REPLACEMENT,
    epoch_count: Annotated[
        int,
        typer.Option("--epoch-count", "-e", help="Number of epochs per match"),
    ] = 30,
    max_simulation_steps: Annotated[
        int,
        typer.Option(
            "--max-simulation-steps",
            help="Maximum number of steps per simulation if no robot wins",
        ),
    ] = 1000,
    record: Annotated[
        bool,
        typer.Option(
            "--record", "-r", help="Define if the simulation is recorded or not"
        ),
    ] = False,
):
    srt.start(
        host,
        port,
        sim_name,
        controllers,
        reward_handler,
        combination_type,
        max_simulation_steps,
        epoch_count,
        record,
    )


@app.command(help="If something has to be tried out: Do it here")
def tryout():
    to.main()


@app.command(help="Runs a gymnasium sample session")
def sample(
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the run")],
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            help="Number of epochs to be run",
        ),
    ] = 100,
    record: Annotated[
        bool,
        typer.Option(
            "--record", "-r", help="Define if the simulation is recorded or not"
        ),
    ] = False,
    host: Annotated[
        str,
        typer.Option(
            "--host", "-h", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port", "-p", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    reward_handler: Annotated[
        sr.RewardHandlerName,
        typer.Option("--reward-handler", help="Name of the reward handler"),
    ] = sr.RewardHandlerName.CONTINUOUS_CONSIDER_ALL,
    opponent: Annotated[
        sr.ControllerName,
        typer.Option("--opponent", help="Name of the opponent controllers"),
    ] = sr.ControllerName.TUMBLR,
):
    sgym_sample.sample(name, epoch_count, record, host, port, opponent, reward_handler)


@app.command(help="Runs a gymnasium q-learning session")
def qtrain(
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the run")],
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            "-e",
            help="Number of epochs to be run",
        ),
    ] = 100,
    host: Annotated[
        str,
        typer.Option(
            "--host", "-h", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port", "-p", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    reward_handler: Annotated[
        sr.RewardHandlerName,
        typer.Option("--reward-handler", help="Name of the reward handler"),
    ] = sr.RewardHandlerName.CONTINUOUS_CONSIDER_ALL,
    opponent: Annotated[
        sr.ControllerName,
        typer.Option("--opponent", help="Name of the opponent controllers"),
    ] = sr.ControllerName.STAND_STILL,
    auto_naming: Annotated[
        bool,
        typer.Option("--auto-naming", help="Create automated unique name"),
    ] = False,
    record: Annotated[
        bool,
        typer.Option(
            "--record", "-r", help="Define if the simulation is recorded or not"
        ),
    ] = False,
    out_dir: Annotated[
        str | None,
        typer.Option("--out-dir", "-o", help="Output directory. Must be absolute"),
    ] = None,
):
    sgym_qlearn.q_train(
        name,
        auto_naming,
        epoch_count,
        host,
        port,
        opponent,
        reward_handler,
        record,
        out_dir,
    )


@app.command(help="Runs cross validation on q-learning session")
def qcv(
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the run")],
    host: Annotated[
        str,
        typer.Option(
            "--host", "-h", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port", "-p", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            "-e",
            help="Number of epochs to be run",
        ),
    ] = 100,
):
    sgym_qlearn.q_train_cv(name, host, port, epoch_count)


@app.command(help="Runs a list of training configurations parallel")
def parallel():
    prl.main()


@app.command(help="Some database management")
def db(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="Name of a query function in module 'simdb'. E.g. 'count_running'",
        ),
    ],
):
    training.simdb(query)


@app.command(help="Runs a gymnasium sample session")
def analysis(
    analysis_name: Annotated[
        an.AnalysisName,
        typer.Option("--analysis", "-a", help="Name of the analysis to be processed"),
    ] = an.AnalysisName.ADJUST_REWARD,
):
    an.main(analysis_name)


@app.command(help="Export simulations from the local database to a file")
def export(
    name: Annotated[
        str,
        typer.Option(
            "--name", "-n", help="Part of the simulations name to be exported"
        ),
    ],
    description: Annotated[
        str,
        typer.Option(
            "--description",
            "-d",
            help="Description of the (behaviour) exported simulations. Use \\n for defining multiline strings",
        ),
    ],
):
    multi_line = description.replace("\\n", "\n")
    exp.export_simulations(name, multi_line)


if __name__ == "__main__":
    app()
