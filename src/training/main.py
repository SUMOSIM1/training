from pathlib import Path

import typer
from typing_extensions import Annotated

import training.reward.reward_core as rhc
import training.sgym.qlearn as sgym_qlearn
import training.sgym.parallel_session as _ps
import training.sgym.sample as sgym_sample
import training.simdb
import training.simrunner as sr
import training.simrunner_tournament as srt
import training.util
import training.tryout as to
import training.explore.analysis as _analysis
import training.explore.export as _export
import training.explore.report as _report
import training.parallel as _parallel

app = typer.Typer(pretty_exceptions_enable=False, add_completion=False)

sim_path_help = "Path to the simulator git module"
sim_path_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"


@app.command(help="Tournament of simulations for combinations of controllers")
def tournament(
    sim_name: Annotated[str, typer.Option("--name", "-n", help="Simulation name")],
    sim_host: Annotated[
        str,
        typer.Option(
            "--sim-host", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    sim_port: Annotated[
        int,
        typer.Option(
            "--sim-port", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    db_host: Annotated[
        str,
        typer.Option("--db-host", help="The host on which the simulation is listening"),
    ] = "localhost",
    db_port: Annotated[
        int,
        typer.Option("--db-port", help="The port on which the simulation is listening"),
    ] = 27017,
    controllers: Annotated[
        list[sr.ControllerName],
        typer.Option("--controllers", "-c", help="Name of controllers"),
    ] = (sr.ControllerName.TUMBLR, sr.ControllerName.BLIND_TUMBLR),
    reward_handler: Annotated[
        rhc.RewardHandlerName,
        typer.Option("--reward-handler", help="Name of the reward handler"),
    ] = rhc.RewardHandlerName.CONTINUOUS_CONSIDER_ALL,
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
    srt.tournament(
        sim_host,
        sim_port,
        db_host,
        db_port,
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


@app.command(help="Gymnasium sample session")
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
    sim_host: Annotated[
        str,
        typer.Option(
            "--sim-host", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    sim_port: Annotated[
        int,
        typer.Option(
            "--sim-port", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    db_host: Annotated[
        str,
        typer.Option("--db-host", help="The host on which the simulation is listening"),
    ] = "localhost",
    db_port: Annotated[
        int,
        typer.Option("--db-port", help="The port on which the simulation is listening"),
    ] = 27017,
    reward_handler: Annotated[
        rhc.RewardHandlerName,
        typer.Option("--reward-handler", help="Name of the reward handler"),
    ] = rhc.RewardHandlerName.CONTINUOUS_CONSIDER_ALL,
    opponent: Annotated[
        sr.ControllerName,
        typer.Option("--opponent", help="Name of the opponent controllers"),
    ] = sr.ControllerName.TUMBLR,
):
    sgym_sample.sample(
        name,
        epoch_count,
        record,
        sim_host,
        sim_port,
        db_host,
        db_port,
        opponent,
        reward_handler,
    )


@app.command(help="Gymnasium q-learning session")
def qlearn(
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the run")],
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            "-e",
            help="Number of epochs to be run",
        ),
    ] = 100,
    sim_host: Annotated[
        str,
        typer.Option(
            "--sim-host", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    sim_port: Annotated[
        int,
        typer.Option(
            "--sim-port", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    db_host: Annotated[
        str,
        typer.Option("--db-host", help="The host on which the simulation is listening"),
    ] = "localhost",
    db_port: Annotated[
        int,
        typer.Option("--db-port", help="The port on which the simulation is listening"),
    ] = 27017,
    record: Annotated[
        bool,
        typer.Option(
            "--record", "-r", help="Define if the simulation is recorded or not"
        ),
    ] = False,
    out_dir: Annotated[
        str,
        typer.Option("--out-dir", "-o", help="Output directory. Must be absolute"),
    ] = "/tmp",
):
    sgym_qlearn.q_learn(
        name=name,
        epoch_count=epoch_count,
        sim_host=sim_host,
        sim_port=sim_port,
        db_host=db_host,
        db_port=db_port,
        record=record,
        out_dir=out_dir,
        q_learn_config=sgym_qlearn.default_q_learn_config,
    )


@app.command(help="Q-learning session. Used by 'parallel'")
def parallel_session(
    name: Annotated[str, typer.Option("--name", help="Name of the run")],
    parallel_config: Annotated[
        _parallel.ParallelConfig,
        typer.Option(
            "--parallel-config",
            help="Parallel configuration",
        ),
    ],
    max_parallel: Annotated[
        int,
        typer.Option(
            "--max-parallel",
            help="Number of maximal executable training processes on the local machine",
        ),
    ],
    parallel_index: Annotated[
        int,
        typer.Option(
            "--parallel-index",
            help="Parallel configuration",
        ),
    ],
    record: Annotated[
        bool,
        typer.Option(
            "--record", "-r", help="Define if the simulation is recorded or not"
        ),
    ] = False,
    sim_host: Annotated[
        str,
        typer.Option(
            "--sim-host", help="The host on which the simulation is listening"
        ),
    ] = "localhost",
    sim_port: Annotated[
        int,
        typer.Option(
            "--sim-port", help="The port on which the simulation is listening"
        ),
    ] = 4444,
    db_host: Annotated[
        str,
        typer.Option("--db-host", help="The host on which the simulation is listening"),
    ] = "localhost",
    db_port: Annotated[
        int,
        typer.Option("--db-port", help="The port on which the simulation is listening"),
    ] = 27017,
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            help="Number of epochs to be run",
        ),
    ] = 100,
    out_dir: Annotated[
        str,
        typer.Option("--out-dir", help="Output directory. Must be absolute"),
    ] = "/tmp",
):
    _ps.parallel_session(
        name=name,
        record=record,
        parallel_config=parallel_config,
        max_parallel=max_parallel,
        parallel_index=parallel_index,
        sim_host=sim_host,
        sim_port=sim_port,
        db_host=db_host,
        db_port=db_port,
        epoch_count=epoch_count,
        out_dir=out_dir,
    )


@app.command(help="List of learning sessions in parallel")
def parallel(
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the run")],
    parallel_config: Annotated[
        _parallel.ParallelConfig,
        typer.Option(
            "--parallel-config",
            help="Parallel configuration",
        ),
    ],
    max_parallel: Annotated[
        int,
        typer.Option(
            "--max-parallel",
            help="Number of maximal executable training processes on the local machine",
        ),
    ],
    parallel_indexes: Annotated[
        str,
        typer.Option(
            "--parallel-indexes",
            "-i",
            help="Comma separated list of indexes or 'all'. E.g. '1,2,3,4', 'all'",
        ),
    ] = "all",
    epoch_count: Annotated[
        int,
        typer.Option(
            "--epoch-count",
            "-e",
            help="Number of epochs to be run",
        ),
    ] = 100,
    db_host: Annotated[
        str,
        typer.Option("--db-host", help="The host on which the simulation is listening"),
    ] = "localhost",
    db_port: Annotated[
        int,
        typer.Option("--db-port", help="The port on which the simulation is listening"),
    ] = 27017,
    keep_container: Annotated[
        bool,
        typer.Option(
            "--keep-container",
            help="Keep container after processing. Useful for error analyse",
        ),
    ] = False,
    record: Annotated[
        bool,
        typer.Option(
            "--record", "-r", help="Define if the simulation is recorded or not"
        ),
    ] = False,
    out_dir: Annotated[
        str,
        typer.Option("--out-dir", "-o", help="Output directory. Must be absolute"),
    ] = "/tmp",
):
    _parallel.parallel_main(
        name=name,
        parallel_config=parallel_config,
        max_parallel=max_parallel,
        parallel_indexes=parallel_indexes,
        epoch_count=epoch_count,
        db_host=db_host,
        db_port=db_port,
        keep_container=keep_container,
        record=record,
        out_dir=out_dir,
    )


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


@app.command(help="Various Analyses")
def analysis(
    analysis_name: Annotated[
        _analysis.AnalysisName,
        typer.Option("--analysis", "-a", help="Name of the analysis to be processed"),
    ] = _analysis.AnalysisName.ADJUST_REWARD,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", "-o", help="Output directory")
    ] = Path.home() / "tmp" / "sumosim" / "analysis",
):
    _analysis.analysis_main(analysis_name, out_dir)


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
    _export.export_simulations(name, multi_line)


@app.command(help="List of learning sessions in parallel")
def report(
    result_dir: Annotated[
        Path,
        typer.Option("--results-dir", "-d", help="Input directory(s) for results"),
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory")],
):
    _report.report(
        result_dir=result_dir,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    app()
