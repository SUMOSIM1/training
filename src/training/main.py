from pathlib import Path

import typer
from typing_extensions import Annotated
import numpy as np

import training.sgym.qlearn as sgym_qlearn
import training.sgym.sample as sgym_sample
import training.simdb
import training.simrunner as sr
import training.simrunner_tournament as srt
import training.util
import training.tryout as to
import training.explore.analysis as an
import training.sgym.core as sgym

app = typer.Typer(pretty_exceptions_enable=False)

sim_path_help = "Path to the simulator git module"
sim_path_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"


@app.command(help="Runs simulations for combinations of controllers")
def start(
    sim_name: Annotated[str, typer.Option("--name", "-n", help="Simulation name")],
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
    sgym_sample.sample(name, epoch_count, record, port, opponent, reward_handler)


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
):
    env_config = sgym.SEnvConfig(
        max_wheel_speed=7,
        wheel_speed_steps=10,
        max_view_distance=700,
        view_distance_steps=3,
        max_simulation_steps=1000,
        dtype=np.float32,
    )
    q_learning_config = sgym_qlearn.QLearnConfig(
        learning_rate=0.01,
        initial_epsilon=0.01,
        epsilon_decay=0.001,
        final_epsilon=0.05,
        discount_factor=0.95,
    )
    sgym_qlearn.q_train(
        name,
        auto_naming,
        epoch_count,
        port,
        opponent,
        reward_handler,
        env_config,
        q_learning_config,
    )


@app.command(help="Runs cross validation on q-learning session")
def qcv(
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the run")],
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            "-e",
            help="Number of epochs to be run",
        ),
    ] = 100,
):
    sgym_qlearn.q_train_cv(name, epoch_count)


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


if __name__ == "__main__":
    app()
