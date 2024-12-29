import subprocess as sp
import socket

from pathlib import Path

_port = 4444


def call(command: list[str]) -> str:
    process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    b_out, b_err = process.communicate()
    if b_err:
        cmd_str = " ".join(command)
        msg = f"ERROR: calling '{cmd_str}'. \n{b_err.decode()}"
        raise RuntimeError(msg)
    return b_out.decode()


def remove_stopped_training_containers() -> str:
    return call(
        [
            "docker",
            "container",
            "ls",
            "-a",
            "-f",
            '"NAME=training-sumo"',
            "-q",
            "|",
            "xargs",
            "docker",
            "container",
            "rm",
        ]
    )


def create_network(name: str) -> str:
    try:
        return call(["docker", "network", "create", name])
    except RuntimeError as er:
        if "exists" not in str(er).lower():
            raise er
        return "<network exists>"


def start_simulator(sim_name: str, network_name: str) -> str:
    try:
        return call(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                sim_name,
                "--network",
                network_name,
                "sumo",
                "sumo",
                "udp",
                "--port",
                str(_port),
            ]
        )
    except RuntimeError as er:
        if "is already in use by container" not in str(er).lower():
            raise er
        return "<already running>"


def start_training(
    training_name: str,
    sim_name: str,
    network_name: str,
    epoch_count: int,
    db_host: str,
    db_port: str,
    out_dir: Path,
) -> str:
    out_dir_str = str(out_dir.absolute())
    user = call(["id", "-u"]).strip()
    group = call(["id", "-g"]).strip()
    db_host_ip = socket.gethostbyname(db_host)
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "-e",
        "PYTHONUNBUFFERED=True",
        "--name",
        training_name,
        "--network",
        network_name,
        "--user",
        f"{user}:{group}",
        "-v",
        f"{out_dir_str}:/tmp",
        "sumot",
        "uv",
        "run",
        "sumot",
        "qtrain",
        "-n",
        "PA",
        "--auto-naming",
        "-e",
        str(epoch_count),
        "--sim-port",
        str(_port),
        "--sim-host",
        sim_name,
        "-o",
        "/tmp/sumosim/q/docker",
        "-r",
        "--db-host",
        db_host_ip,
        "--db-port",
        str(db_port),
    ]
    print(f"Start training using: '{' '.join(cmd)}'")
    return call(cmd)


def start_training_configuration(
    nr: int, epoch_count: int, db_host: str, db_port: int, out_dir: Path
):
    out_dir.mkdir(exist_ok=True, parents=True)
    network_name = f"sumo{nr:02d}"
    sim_name = f"sumo{nr:02d}"
    training_name = f"sumo-train{nr:02d}"

    network_id = create_network(network_name)
    print(f"Created network {network_name} {network_id}")

    sim_run_id = start_simulator(sim_name, network_name)
    print(f"Started simulator {sim_name} {sim_run_id}")

    training_run_id = start_training(
        training_name, sim_name, network_name, epoch_count, db_host, db_port, out_dir
    )
    print(f"Started training {training_name} {training_run_id}")


def main(
    epoch_count: int, config_count: int, db_host: str, db_port: int, out_dir: str | None
):
    print("Started parallel")
    out_path = Path.home() / "tmp"
    if out_dir:
        out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    for nr in range(config_count):
        start_training_configuration(nr, epoch_count, db_host, db_port, out_path)
