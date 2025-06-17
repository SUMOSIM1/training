import socket
import itertools as it
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import training.helper as hlp
from typing import cast


_port = 4444


class ParallelConfig(Enum):
    Q_CROSS_0 = "q-cross-0"
    Q_CROSS_1 = "q-cross-1"
    Q_MAP_0 = "q-map-0"
    Q_MAP_1 = "q-map-1"
    Q_MAP_2 = "q-map-2"
    Q_MAP_3 = "q-map-3"
    Q_MAP_4 = "q-map-4"
    Q_RW_0 = "q-rw-0"
    Q_RW_1 = "q-rw-1"
    Q_RW_2 = "q-rw-2"
    Q_RW_3 = "q-rw-3"
    Q_RW_4 = "q-rw-4"
    Q_LOW_0 = "q-low-0"
    Q_LOW_1 = "q-low-1"
    Q_LOW_2 = "q-low-2"
    Q_SEE_0 = "q-see-0"
    Q_SEE_1 = "q-see-1"
    Q_SEE_2 = "q-see-2"
    Q_FETCH_0 = "q-fetch-0"
    Q_FETCH_1 = "q-fetch-1"
    Q_FETCH_2 = "q-fetch-2"
    Q_ED_1 = "q-ed-1"
    Q_ED_2 = "q-ed-2"
    Q_ED_3 = "q-ed-3"
    Q_EDEXP_1 = "q-edexp-1"
    Q_EDC_1 = "q-edc-1"
    Q_EDC_2 = "q-edc-2"
    Q1_TEST_1 = "q1-test-1"


@dataclass(frozen=True)
class ParallelSessionConfig:
    name: str
    values: dict


def create_parallel_session_configs(
    parallel_config: ParallelConfig, max_parallel: int
) -> list[list[ParallelSessionConfig]]:
    match parallel_config:
        case ParallelConfig.Q_CROSS_0:
            values_dict = {
                "L": [0.5, 0.7, 0.1, 0.2],
                "E": [0.5, 0.7, 0.1, 0.2],
                "D": [0.2, 0.7, 0.8, 0.9, 0.99],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_CROSS_1:
            values_dict = {
                "L": [0.7, 0.8, 0.9],
                "E": [0.01, 0.05, 0.1],
                "D": [0.95, 0.99, 0.995, 0.999],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_MAP_0:
            values_dict = {
                "L": [0.7],
                "E": [0.05],
                "D": [0.5, 0.8],
                "M": ["non-linear-1", "non-linear-2", "non-linear-3", "non-linear-4"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_MAP_1:
            values_dict = {
                "L": [0.01, 0.1, 0.5],
                "E": [0.01, 0.1],
                "D": [0.4, 0.6, 0.9],
                "M": ["non-linear-1", "non-linear-2"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_MAP_2:
            values_dict = {
                "L": [0.2, 0.4, 0.6],
                "E": [0.05, 0.075, 0.1],
                "D": [0.3, 0.4, 0.5],
                "M": ["non-linear-1", "non-linear-2", "non-linear-3", "non-linear-4"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_MAP_3:
            values_dict = {
                "L": [0.05, 0.1, 0.15],
                "E": [0.01, 0.02, 0.03],
                "D": [0.25, 0.3, 0.35],
                "M": ["non-linear-2", "non-linear-3", "non-linear-4"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_MAP_4:
            values_dict = {
                "L": [0.12, 0.12, 0.12],
                "E": [0.015, 0.015, 0.015],
                "D": [0.3, 0.3, 0.3],
                "M": ["non-linear-3", "non-linear-3", "non-linear-3"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_RW_0:
            values_dict = {
                "L": [0.12, 0.12],
                "E": [0.015, 0.015],
                "D": [0.3, 0.3],
                "M": ["non-linear-3", "non-linear-3"],
                "R": ["continuous-consider-all", "reduced-push-reward"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_RW_1:
            values_dict = {
                "L": [0.12, 0.12],
                "E": [0.015, 0.015],
                "D": [0.3, 0.3],
                "M": ["non-linear-3", "non-linear-3"],
                "R": ["continuous-consider-all", "reduced-push-reward", "speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_RW_2:
            values_dict = {
                "L": [0.15, 0.1, 0.05],
                "E": [0.015, 0.01, 0.005],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["speed-bonus", "speed-bonus", "speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_RW_3:
            values_dict = {
                "L": [0.25, 0.2, 0.15],
                "E": [0.025, 0.02, 0.015],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["speed-bonus", "speed-bonus", "speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_LOW_0:
            values_dict = {
                "L": [0.001, 0.0001],
                "E": [0.001, 0.0001],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["speed-bonus", "speed-bonus", "speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_LOW_1:
            values_dict = {
                "L": [0.01, 0.005, 0.001],
                "E": [0.01, 0.005, 0.001],
                "D": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                "M": ["non-linear-3"],
                "R": ["speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_LOW_2:
            values_dict = {
                "L": [0.01, 0.005, 0.001],
                "E": [0.01, 0.005, 0.001],
                "D": [0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8],
                "M": ["non-linear-3"],
                "R": ["speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)

        case ParallelConfig.Q_SEE_0:
            values_dict = {
                "L": [0.2, 0.2, 0.2, 0.2, 0.2],
                "E": [0.02, 0.02, 0.02, 0.02, 0.02],
                "D": [0.3, 0.3, 0.3, 0.3],
                "M": ["non-linear-3"],
                "R": ["speed-bonus"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_SEE_1:
            values_dict = {
                "L": [0.2, 0.2, 0.2, 0.2, 0.2],
                "E": [0.02, 0.02, 0.02, 0.02, 0.02],
                "D": [0.3, 0.3, 0.3, 0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_SEE_2:
            values_dict = {
                "L": [0.2, 0.2, 0.2, 0.2, 0.2],
                "E": [0.01, 0.01, 0.01, 0.01, 0.01],
                "D": [0.3, 0.3, 0.3, 0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_FETCH_0:
            values_dict = {
                "L": [0.2, 0.2, 0.2],
                "E": [0.01],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["lazy-s", "lazy-m", "lazy-l"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_FETCH_1:
            values_dict = {
                "L": [0.2, 0.2, 0.2],
                "E": [0.01, 0.01, 0.01],
                "D": [0.3, 0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["lazy-s", "lazy-m", "lazy-l"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_FETCH_2:
            values_dict = {
                "L": [0.2, 0.2],
                "E": [0.01, 0.01],
                "D": [0.3, 0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": [
                    "lazy-s-t2",
                    "lazy-s-t5",
                    "lazy-s-t10",
                    "lazy-m-t2",
                    "lazy-m-t5",
                    "lazy-m-t10",
                    "lazy-l-t2",
                    "lazy-l-t5",
                    "lazy-l-t10",
                ],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_ED_1:
            values_dict = {
                "L": [0.2, 0.2, 0.2],
                "E": [0.05],
                "ED": ["none", "decay-100-80", "decay-100-50", "decay-100-20"],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_ED_2:
            values_dict = {
                "L": [0.2, 0.2, 0.2, 0.2],
                "E": [0.05],
                "ED": [
                    "none",
                    "decay-1000-80",
                    "decay-1000-50",
                    "decay-1000-20",
                    "none",
                    "decay-3000-80",
                    "decay-3000-50",
                    "decay-3000-20",
                ],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_ED_3:
            values_dict = {
                "L": [0.2, 0.2, 0.2, 0.2],
                "E": [0.05],
                "ED": [
                    "decay-1000-20",
                    "decay-1000-10",
                    "decay-1000-05",
                    "decay-3000-20",
                    "decay-3000-10",
                    "decay-3000-05",
                ],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_EDEXP_1:
            values_dict = {
                "L": [0.2, 0.2, 0.2, 0.2],
                "E": [0.05],
                "ED": [
                    "none",
                    "decay-exp-100",
                    "decay-exp-1000",
                    "decay-exp-5000",
                ],
                "D": [0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_EDC_1:
            values_dict = {
                "L": [0.01, 0.1, 0.2, 0.5],
                "E": [0.01, 0.05, 0.1, 0.5],
                "ED": ["decay-exp-1000"],
                "D": [0.2, 0.3, 0.5, 0.8],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_EDC_2:
            values_dict = {
                "L": [0.1, 0.5, 0.8, 1.0],
                "E": [0.01, 0.05, 0.1, 0.5],
                "ED": ["decay-exp-1000"],
                "D": [0.05, 0.1, 0.2, 0.3],
                "M": ["non-linear-3"],
                "R": ["can-see"],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q_RW_4:
            values_dict = {
                "L": [0.8, 0.8, 0.8, 0.8],
                "E": [0.08],
                "ED": ["decay-exp-1000"],
                "D": [0.25],
                "M": ["non-linear-3"],
                "R": [
                    "continuous-consider-all",
                    "reduced-push-reward",
                    "speed-bonus",
                    "can-see",
                ],
                "F": ["eager"],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case ParallelConfig.Q1_TEST_1:
            values_dict = {
                "L": [0.8, 0.8, 0.8, 0.8],
            }
            return _create_parallel_session_configs(values_dict, max_parallel)
        case _:
            raise ValueError(f"Invalid Parallel Config {parallel_config.value}")


def _create_parallel_session_configs(
    values_dict: dict, max_parallel: int
) -> list[list[ParallelSessionConfig]]:
    def to_dict(keys: list, batch_values: list) -> list[dict]:
        return [dict(zip(keys, value)) for value in batch_values]

    def to_ids(dicts: list[dict]) -> list[str]:
        def to_id(dictionary: dict) -> str:
            return "".join([f"{key}{index}" for key, index in dictionary.items()])

        return [to_id(dictionary) for dictionary in dicts]

    def create_batched_dicts(keys: list, lists: list) -> list[list[dict]]:
        prod_values = list(it.product(*lists))
        batch_size = (len(prod_values) + max_parallel - 1) // max_parallel
        batched_values = it.batched(prod_values, batch_size)
        return [to_dict(keys, cast(list, values)) for values in batched_values]

    _keys = list(values_dict.keys())
    _values = [values_dict[k] for k in _keys]
    batched_dicts = create_batched_dicts(_keys, _values)

    index_values = [range(len(list(value))) for value in _values]
    batched_index_dicts = create_batched_dicts(_keys, index_values)
    ids = [to_ids(d) for d in batched_index_dicts]

    double_zipped = [zip(a, b) for a, b in (zip(batched_dicts, ids))]
    return [
        [ParallelSessionConfig(values=values, name=name) for values, name in zipped]
        for zipped in double_zipped
    ]


def create_network(name: str) -> str:
    try:
        return hlp.call(["docker", "network", "create", name])
    except RuntimeError as er:
        if "exists" not in str(er).lower():
            raise er
        return "<network exists>"


def start_simulator(sim_name: str, network_name: str) -> str:
    try:
        return hlp.call(
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


def parse_parallel_indexes(
    parallel_indexes: str, parallel_config: ParallelConfig, max_parallel: int
) -> list[int]:
    configs = create_parallel_session_configs(parallel_config, max_parallel)
    if parallel_indexes.lower() == "all":
        return list(range(len(configs)))
    indexes = hlp.parse_integers(parallel_indexes)
    max_index = len(configs) - 1
    for index in indexes:
        if index > max_index:
            raise ValueError(
                f"ERROR: Cannot start simulation for parallel index {index} "
                f"of {parallel_config.value} "
                f"with max_parallel {max_parallel}. Max index is {max_index}"
            )
    return indexes


def start_training(
    name: str,
    parallel_config: ParallelConfig,
    max_parallel: int,
    parallel_index: int,
    sim_name: str,
    network_name: str,
    epoch_count: int,
    db_host: str,
    db_port: str,
    keep_container: bool,
    record: bool,
    out_dir: Path,
) -> str:
    out_dir_str = str(out_dir.absolute())
    user = hlp.call(["id", "-u"]).strip()
    group = hlp.call(["id", "-g"]).strip()
    db_host_ip = socket.gethostbyname(db_host)
    train_name = f"sumo-train{parallel_index:02d}"
    cmd = [
        "docker",
        "run",
        "-d",
        None if keep_container else "--rm",
        "-e",
        "PYTHONUNBUFFERED=True",
        "--name",
        train_name,
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
        "parallel-session",
        "--name",
        name,
        "--record" if record else None,
        "--parallel-config",
        parallel_config.value,
        "--max-parallel",
        str(max_parallel),
        "--parallel-index",
        str(parallel_index),
        "--epoch-count",
        str(epoch_count),
        "--sim-port",
        str(_port),
        "--sim-host",
        sim_name,
        "--out-dir",
        "/tmp",
        "--db-host",
        db_host_ip,
        "--db-port",
        str(db_port),
    ]
    cmd = [x for x in cmd if x is not None]
    print(f"Start training using: '{' '.join(cmd)}'")
    return hlp.call(cmd)


def start_training_configuration(
    name: str,
    parallel_config: ParallelConfig,
    max_parallel: int,
    parallel_index: int,
    epoch_count: int,
    db_host: str,
    db_port: int,
    keep_container: bool,
    record: bool,
    out_dir: Path,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    network_name = f"sumo{parallel_index:02d}"
    sim_name = f"sumo-sim{parallel_index:02d}"

    network_id = create_network(network_name)
    print(f"Created network {network_name} {network_id}")

    sim_run_id = start_simulator(sim_name, network_name)
    print(f"Started simulator {sim_name} {sim_run_id}")

    training_run_id = start_training(
        name=name,
        parallel_config=parallel_config,
        max_parallel=max_parallel,
        parallel_index=parallel_index,
        sim_name=sim_name,
        network_name=network_name,
        epoch_count=epoch_count,
        db_host=db_host,
        db_port=db_port,
        keep_container=keep_container,
        record=record,
        out_dir=out_dir,
    )
    print(f"Started training {name} {training_run_id}")


def subdir_exists(work_dir: Path, prefix: str) -> bool:
    for x in work_dir.iterdir():
        if x.is_dir() and x.name.startswith(prefix):
            return True
    return False


def parallel_main(
    name: str,
    parallel_config: ParallelConfig,
    max_parallel: int,
    parallel_indexes: str,
    epoch_count: int,
    db_host: str,
    db_port: int,
    keep_container: bool,
    record: bool,
    out_dir: str,
):
    print("Started parallel")
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    if subdir_exists(out_path, name):
        raise RuntimeError(f"Output directory '{out_path}' {name} already exists")
    for parallel_index in parse_parallel_indexes(
        parallel_indexes, parallel_config, max_parallel
    ):
        start_training_configuration(
            name,
            parallel_config,
            max_parallel,
            parallel_index,
            int(epoch_count),
            db_host,
            db_port,
            keep_container,
            record,
            out_path,
        )
