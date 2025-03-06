"""
Functions for creating, dump/load, manipulate q-tables
"""


import training.sgym.core as sgym


def all_obs(q_learn_env_config: sgym.SEnvConfig) -> list[any]:
    all = []
    for opponent_see in range(q_learn_env_config.opponent_see_steps):
        for left_see in range(q_learn_env_config.view_distance_steps):
            for front_see in range(q_learn_env_config.view_distance_steps):
                for right_see in range(q_learn_env_config.view_distance_steps):
                    all.append((opponent_see, left_see, front_see, right_see))
    return all


def from_key(key: tuple[int, int, int, int]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"


def to_key(strval: str) -> tuple[int, int, int, int]:
    split = strval.split("|")
    return (int(split[0]), int(split[1]), int(split[2]), int(split[3]))


def to_json(q_table: dict, q_table_keys: list[tuple]) -> dict:
    json_dict = {}
    for k in q_table_keys:
        json_dict[from_key(k)] = q_table[k]
    return json_dict
