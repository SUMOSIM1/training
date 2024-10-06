import json
import random
import subprocess
import uuid
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt

import training.reward as rw
import training.simdb as sd
import training.simrunner as sr


def util_sims_to_json_files(sim_names: list[str], base_dir: Path) -> list[Path]:
    out_files = []
    with sd.create_client() as c:
        all_sims = sd.find_all(c)
        for sim in all_sims:
            if sim["name"] in sim_names:
                of = util_sim_to_json_file(sim, base_dir)
                out_files.append(of)
    return out_files


def parse_sim_winner(events: dict) -> int:
    """
    :param events: dict containing winner info
        {'r1': [['draw', 'true']], 'r2': [['draw', 'true']]} draw => 0
        {'r1': [['winner', 'true']], 'r2': []} robot1 winner => 1
        {'r1': [], 'r2': [['winner', 'true']]} robot2 winner => 2
    :return: 0: draw, 1: robot1 winner, 2: robot2 winner
    """
    if events["r1"]:
        if events["r1"][0][0] == "draw":
            return 0
        elif events["r1"][0][0] == "winner":
            return 1
    return 2


def util_sim_to_json_file(db_sim: dict, base_dir: Path) -> Path:
    n = db_sim["name"]
    n1 = db_sim["robot1"]["name"]
    n2 = db_sim["robot2"]["name"]
    sim_name = f"{n}_{n1}_{n2}"
    file_name = f"{sim_name}.json"
    sim_path = base_dir / file_name
    with sim_path.open("w", encoding="UTF-8") as f:
        sim_save_dict = {
            "name": sim_name,
            "winner": db_sim["events"],
            "states": db_sim["states"],
        }
        json.dump(sim_save_dict, f, indent=2)
        # print(f"### wrote {sim_name} to {sim_path}")
    return sim_path


def read_file(file: Path) -> rw.Sim:
    with file.open() as f:
        data_dict = json.load(f)
        states = data_dict["states"]
        states1 = [rw.RobotsState.from_dict(s) for s in states]
        return rw.Sim(
            name=data_dict["name"],
            winner=parse_sim_winner(data_dict["winner"]),
            states=states1,
        )


def can_see_bool(r1: rw.RobotState, r2: rw.RobotState, h: float) -> float | None:
    v = rw.can_see(r1, r2)
    if v and v < 200:
        return h
    return None


def print_distances(file: Path):
    sim = read_file(file)
    dists = [rw.dist(s) for s in sim.states]
    for d in dists:
        if d < 101:
            print(d)


def util_visualize(files: list[Path], run_name: str, n: int, m: int, out_dir: Path):
    lineWidth = 5.0
    scale = 2.0
    w = 11.69 * scale
    h = 8.25 * scale
    fig = plt.figure(figsize=(w, h), facecolor="white")
    for k, file in enumerate(files):
        sim = read_file(file)
        dists = [rw.dist(s) for s in sim.states]
        # pprint(dists)
        r1_can_see = [can_see_bool(s.robot1, s.robot2, 150) for s in sim.states]
        r2_can_see = [can_see_bool(s.robot2, s.robot1, 130) for s in sim.states]
        ax = fig.add_subplot(m, n, k + 1)
        ax.plot(
            range(len(dists)),
            dists,
            color="tab:blue",
            label="distance",
            dashes=[4, 1],
            linewidth=2.0,
        )
        ax.plot(
            range(len(r1_can_see)),
            r1_can_see,
            color="tab:red",
            label="r1 see opp.",
            linewidth=lineWidth,
        )
        ax.plot(
            range(len(r2_can_see)),
            r2_can_see,
            color="tab:orange",
            label="r2 see opp.",
            linewidth=lineWidth,
        )
        ax.plot(
            [],
            [],
            " ",
            label=f"winner is {sim.winner}",
        )
        ax.legend()
        ax.set_title(sim.name)
    if not out_dir.exists():
        out_dir.mkdir()
    file_name = f"{run_name}.pdf"
    file_path = out_dir / file_name
    fig.savefig(file_path, dpi=300)
    print(f"saved color image to {file_path}")
    file_name_gray = f"{run_name}-gray.pdf"
    file_path_gray = out_dir / file_name_gray
    subprocess.run(["convert", "-grayscale", "average", file_path, file_path_gray])
    print(f"saved grayscale image to {file_path}")


def visualize_from_files(id: str, in_dir: Path, n: int, m: int, out_dir: Path):
    def match(file: Path):
        return file.is_file() and file.name.find(id) >= 0 and file.suffix == ".json"

    files = [f for f in in_dir.iterdir() if match(f)]
    sorted_files = sorted(files, key=lambda f: f.stem)
    print(f"### found {len(sorted_files)} files for id: {id}")
    if n * m != len(sorted_files):
        raise (ValueError(f"Found {len(sorted_files)} for n, m: {n}, {m}"))
    util_visualize(sorted_files, id, n, m, out_dir)


def util_create_simruns(
    n: int, m: int, host: str, id: str, base_dir: Path, out_dir: Path
):
    def create(i: int, run_name: str) -> str:
        sim_name = f"{run_name}-{i:02d}"
        c1 = random.choice([sr.ControllerName.TUMBLR, sr.ControllerName.STAY_IN_FIELD])
        c2 = random.choice(
            [sr.ControllerName.BLIND_TUMBLR, sr.ControllerName.STAY_IN_FIELD]
        )
        sr.start(
            port=4444,
            sim_name=sim_name,
            controller_name1=c1,
            controller_name2=c2,
        )
        sleep(0.5)
        return sim_name

    n1 = n * m
    run_id = str(uuid.uuid4())[0:6]
    run_name = f"{host}-{id}-{run_id}"
    names = []
    for i in range(n1):
        nam = create(i, run_name)
        names.append(nam)
    print(f"## created {n} simulations. run_name: {run_name}")
    out_files = util_sims_to_json_files(sim_names=names, base_dir=base_dir)
    util_visualize(files=out_files, run_name=run_name, n=n, m=m, out_dir=out_dir)