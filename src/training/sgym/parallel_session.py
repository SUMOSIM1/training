import training.parallel as _p
import training.sgym.qlearn as _ql
from dataclasses import replace


def parallel_session(
    name: str,
    record: bool,
    parallel_config: _p.ParallelConfig,
    max_parallel: int,
    parallel_index: int,
    sim_host: str,
    sim_port: int,
    db_host: str,
    db_port: int,
    epoch_count: int,
    out_dir: str,
):
    def parallel_to_qlearn_config(
        parallel_session_config: _p.ParallelSessionConfig,
    ) -> _ql.QLearnConfig:
        q_learn_config = _ql.default_q_learn_config
        parallel_config_values: dict = parallel_session_config.values
        if parallel_config_values.get("L") is not None:
            q_learn_config = replace(
                q_learn_config, learning_rate=parallel_config_values["L"]
            )
        if parallel_config_values.get("E") is not None:
            q_learn_config = replace(
                q_learn_config,
                epsilon=parallel_config_values["E"],
            )
        if parallel_config_values.get("ED") is not None:
            q_learn_config = replace(
                q_learn_config,
                epsilon_decay_name=parallel_config_values["ED"],
            )
        if parallel_config_values.get("D") is not None:
            q_learn_config = replace(
                q_learn_config, discount_factor=parallel_config_values["D"]
            )
        if parallel_config_values.get("M") is not None:
            q_learn_config = replace(
                q_learn_config,
                mapping_name=parallel_config_values["M"],
            )
        if parallel_config_values.get("R") is not None:
            q_learn_config = replace(
                q_learn_config,
                reward_handler_name=parallel_config_values["R"],
            )
        if parallel_config_values.get("F") is not None:
            q_learn_config = replace(
                q_learn_config,
                fetch_type=parallel_config_values["F"],
            )
        return q_learn_config

    def call_q_learn(parallel_train_config: _p.ParallelSessionConfig):
        q_learn_config = parallel_to_qlearn_config(parallel_train_config)
        _ql.q_learn(
            name=f"{name}-{parallel_train_config.name}",
            epoch_count=epoch_count,
            sim_host=sim_host,
            sim_port=sim_port,
            db_host=db_host,
            db_port=db_port,
            record=record,
            out_dir=out_dir,
            q_learn_config=q_learn_config,
        )

    configs = _p.create_parallel_session_configs(parallel_config, max_parallel)
    if parallel_index >= len(configs):
        raise ValueError(
            f"Cannot run 'parallel_session' because the parallel index {parallel_index} exceeds the maximum index for parallel_config {parallel_config.value}. Max index is {len(configs) - 1}"
        )
    _configs: list[_p.ParallelSessionConfig] = configs[parallel_index]
    for c in _configs:
        call_q_learn(c)
    print(f"Finished parallel session n:{name}")
