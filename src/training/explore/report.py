from pathlib import Path

import yaml
import dominate
import dominate.tags as dt
import dominate.util as du
import shutil
from dataclasses import dataclass
import training.helper as hlp
import training.parallel as pa
import training.explore.enumdescs as edesc
import re

from dominate.dom_tag import dom_tag

from training.explore.allvideos import AllVideoNames


@dataclass(frozen=True)
class Resources:
    videos: list[Path]
    q_values: list[Path]
    boxplots: list[Path]


@dataclass
class CombiLinks1:
    text: str
    boxplots: list[str]
    q_values: list[str]
    videos: list[str]


@dataclass
class CombiLinks:
    prefix: str
    combi: str
    text: str
    boxplots: list[str]
    q_values: list[str]
    videos: list[str]


def extract_combis(lead_resources: list[Path], prefix: str) -> list[str]:
    keys_set = set()
    for resource in lead_resources:
        if resource.name.startswith(prefix):
            name_rest = resource.stem[len(prefix) + 1 :]
            split_name = name_rest.split("-")
            keys_set.add(split_name[0])
    return sorted(list(keys_set))


def extract_combis_from_enum(report_data: dict) -> list[str]:
    enum_names = [
        enum_name
        for enum_name in report_data["enumdescs"]
        if enum_name.startswith("training.parallel.ParallelConfig")
    ]
    combi_keys = []
    if enum_names:
        pc_name = enum_names[0].split(".")[3]
        values = [x.value for x in pa.ParallelConfig]
        # print(pc_name, values)
        if pc_name in values:
            pc = pa.ParallelConfig(pc_name)
            a = pa.create_parallel_session_configs(pc, 10)
            combi_keys = []
            for b in a:
                for c in b:
                    combi_keys.append(c.name)
            combi_keys = sorted(list(combi_keys))
    return combi_keys


def collect_resources(results_dirs: list[Path], result_name: str) -> list[Path]:
    out = []
    for result_path in results_dirs:
        out += [
            r.absolute() for r in result_path.iterdir() if r.stem.endswith(result_name)
        ]
    return sorted(out)


def create_report(
    reports_data: dict,
    result_dir_paths: list[Path],
    out_path: Path,
    all_video_names: AllVideoNames,
):
    videos = collect_resources(result_dir_paths, "sumosim-video")
    q_values = collect_resources(result_dir_paths, "q-values-heat")
    boxplots = collect_resources(result_dir_paths, "boxplot")
    resources = Resources(videos=videos, q_values=q_values, boxplots=boxplots)
    # print(f"### resources v:{len(resources.videos)}")
    # print(f"### resources qv:{len(resources.q_values)}")
    # print(f"### resources b:{len(resources.boxplots)}")

    # Copy style .css
    style_path = Path(__file__).parent.parent.parent.parent / "resources" / "styles.css"
    style_target = out_path / "styles.css"
    shutil.copy(style_path, style_target)
    # print(f"Created style {style_target}")

    # Copy analyses
    analysis_path = (
        Path(__file__).parent.parent.parent.parent / "resources" / "analysis"
    )
    analysis_target = out_path / "analysis"
    shutil.copytree(analysis_path, analysis_target, dirs_exist_ok=True)

    create_report_index(reports_data, out_path, resources, all_video_names)

    print(f"Created reports in {out_path.absolute()}")


# noinspection DuplicatedCode
def create_report_index(
    report_dict: dict,
    out_path: Path,
    resources: Resources,
    all_video_names: AllVideoNames,
):
    out_file = out_path / "index.html"
    method_tuples = [
        create_report_method(method_dict, i, out_path, resources, all_video_names)
        for i, method_dict in enumerate(report_dict["methods"])
    ]
    doc = dominate.document(title=report_dict["title"])
    with doc.head:
        dt.meta(name="viewport", content="width=device-width")
        dt.link(rel="stylesheet", href="styles.css")
    with doc:
        dt.h1().add(report_dict["title"])
        dt.p().add(du.raw(hlp.parse_markdown(report_dict["description"])))
        for text, link in method_tuples:
            dt.a(text, href=link)
            dt.br()
    with out_file.open("w") as f:
        f.write(str(doc))


# noinspection DuplicatedCode
def create_report_method(
    method_dict: dict,
    index: int,
    out_path: Path,
    resources: Resources,
    all_video_names: AllVideoNames,
) -> tuple[str, str]:
    out_file_name = f"method-{index:02d}.html"
    out_file = out_path / out_file_name
    training_tuples = [
        create_report_training(method_dict, i, out_path, resources, all_video_names)
        for i, method_dict in enumerate(method_dict["trainings"])
    ]

    doc = dominate.document(title=method_dict["title"])

    with doc.head:
        dt.meta(name="viewport", content="width=device-width")
        dt.link(rel="stylesheet", href="styles.css")

    with doc:
        dt.h1().add(method_dict["title"])
        dt.p().add(du.raw(hlp.parse_markdown(method_dict["description"])))
        for training_tuple in training_tuples:
            if training_tuple is not None:
                text, link = training_tuple
                dt.a(text, href=link)
                dt.br()

    with out_file.open("w") as f:
        f.write(str(doc))
    return method_dict["title"], out_file_name


def create_report_training(
    training_dict: dict,
    index: int,
    out_path: Path,
    resources: Resources,
    all_video_names: AllVideoNames,
) -> tuple[str, str] | None:
    prefix = training_dict["prefix"]
    color = training_dict["color"]

    def tags_for_enumdescs(training_dict: dict) -> dom_tag:
        enum_keys = training_dict.get("enumdescs")
        if enum_keys is None:
            return dt.div()
        enum_texts = edesc.extract_enumdescs(enum_keys)
        # print("### enum_texts", enum_texts)
        return dt.p(
            [
                (dt.h3(key), du.raw(hlp.parse_markdown(txt)))
                for key, txt in zip(enum_keys, enum_texts)
            ]
        )

    def tags_for_combis(
        resources: Resources, combis: list[CombiLinks], out_path: Path
    ) -> dom_tag:
        def match_resource_name(name: str, combi: str) -> bool:
            name_rest = name[len(prefix) + 1 :]
            split_name = name_rest.split("-")
            return name.startswith(prefix) and split_name[0] == combi

        def filter_and_sort_videos(combi: str) -> list[Path]:
            _videos = [
                _res
                for _res in resources.videos
                if match_resource_name(_res.name, combi)
            ]
            return sorted(_videos)

        def tags_for_video_links(index: int, link: str) -> dom_tag:
            if index > 0 and index % 4 == 0:
                return (dt.br(), dt.a(f"video {index}", href=link))
            return dt.a(f"video {index}", href=link)

        def tags_for_resource(combi_links: CombiLinks) -> dom_tag:
            # print(f"#### tags_for_resource {combi_links.text} {len(combi_links.videos)}")
            return dt.div(
                combi_links.combi,
                dt.br(),
                [
                    dt.a(dt.img(src=link, width=400), href=link)
                    for link in combi_links.boxplots
                ],
                dt.br(),
                [dt.a("q-values", href=link) for link in combi_links.q_values],
                dt.br(),
                [
                    tags_for_video_links(i, link)
                    for i, link in enumerate(combi_links.videos)
                ],
                _class="box",
                _style=f"background-color:{color}",
            )

        return dt.div(
            [tags_for_resource(combi_link) for combi_link in combi_links],
            _class="container",
        )

    def filter_combi_resources(
        resources: Resources, prefix: str, combi: str
    ) -> Resources | None:
        def match_resource_name(name: str) -> bool:
            name_rest = name[len(prefix) + 1 :]
            split_name = name_rest.split("-")
            return name.startswith(prefix) and split_name[0] == combi

        def filter_and_sort_videos() -> list[Path]:
            _videos = [
                _res for _res in resources.videos if match_resource_name(_res.name)
            ]
            return sorted(_videos)

        boxplots = [res for res in resources.boxplots if match_resource_name(res.name)]
        if not boxplots:
            return None
        q_values = [res for res in resources.q_values if match_resource_name(res.name)]
        _videos = filter_and_sort_videos()
        # print(f"-- filter {prefix} {combi}")
        # pp.pprint(boxplots)
        # pp.pprint(_videos)
        return Resources(boxplots=boxplots, q_values=q_values, videos=_videos)

    def copy_missing_resources(resources: Resources, out_path: Path):
        def cp(res_list: list[Path]):
            res_path = out_path / prefix
            res_path.mkdir(parents=True, exist_ok=True)
            for res in res_list:
                target_path = res_path / res.name
                if not target_path.exists():
                    shutil.copy(res, target_path)
                    print(f"Copied {res.name} to {res_path}")

        cp(resources.boxplots)
        cp(resources.q_values)
        cp(resources.videos)

    def resource_to_url(res: Path) -> str:
        return f"{prefix}/{res.name}"

    out_file_name = f"training-{index:02d}.html"
    out_file = out_path / out_file_name
    _combis = extract_combis_from_enum(training_dict)

    prefix = training_dict["prefix"]
    if not _combis:
        print(f"Found no combis for {prefix}")
        return None

    combi_resources = [
        cr
        for cr in [filter_combi_resources(resources, prefix, c) for c in _combis]
        if cr is not None
    ]
    if combi_resources:

        def create_combi_link(combi: str, _resources: Resources) -> CombiLinks:
            return CombiLinks(
                prefix=prefix,
                combi=combi,
                text=training_dict["title"],
                boxplots=[resource_to_url(_res) for _res in _resources.boxplots],
                q_values=[resource_to_url(_res) for _res in _resources.q_values],
                videos=[resource_to_url(_res) for _res in _resources.videos],
            )

        print(f"Found combi resources for {prefix}")
        for res in combi_resources:
            copy_missing_resources(res, out_path)
        print(f"Copied missing combi resources for {prefix}")
        combi_links = [
            create_combi_link(c, res) for c, res in zip(_combis, combi_resources)
        ]
    else:

        def create_combi_link(combi: str) -> CombiLinks | None:
            key = AllVideoNames.create_key(prefix, combi)
            video_names = all_video_names.video_names(key)
            if video_names is None:
                return None
            return CombiLinks(
                prefix=prefix,
                combi=combi,
                text=training_dict["title"],
                boxplots=[f"{prefix}/{prefix}-{combi}-boxplot.png"],
                q_values=[f"{prefix}/{prefix}-{combi}-q-values-heat.mp4"],
                videos=video_names,
            )

        print(f"Create report links for {prefix} using AllVideoNames")
        combi_links = [
            combi_link
            for combi_link in [
                create_combi_link(_combi_link) for _combi_link in _combis
            ]
            if combi_link is not None
        ]

    doc = dominate.document(title=training_dict["title"])

    with doc.head:
        dt.meta(name="viewport", content="width=device-width")
        dt.link(rel="stylesheet", href="styles.css")

    with doc.body:
        dt.h1().add(f"{prefix} {training_dict['title']}")
        dt.p().add(du.raw(hlp.parse_markdown(training_dict["description"].strip())))
        tags_for_enumdescs(training_dict)
        tags_for_combis(resources, combi_links, out_path)

    with out_file.open("w") as f:
        f.write(str(doc))
    return f"{prefix} {training_dict['title']}", out_file_name


def create_final_resources(reports_data: dict, result_dir_paths: list[Path]):
    def create(result_dir_path: Path, prefix: str):
        all_resources = list(
            [r for r in result_dir_path.iterdir() if r.stem.startswith(prefix)]
        )
        for r in all_resources:
            if r.name.startswith(prefix) and r.name.endswith("mp4"):
                pass
                # print("### ", r.name)
        lead_resources = [r for r in all_resources if r.stem.endswith("boxplot")]
        if lead_resources:
            combis = extract_combis(lead_resources, prefix)
            for combi in combis:
                create_heat_video(all_resources, combi, prefix, result_dir_path)
                create_simulation_video(all_resources, combi, prefix, result_dir_path)

    def create_heat_video(all_resources, combi, prefix, result_dir_path):
        name = f"{prefix}-{combi}-q-values-heat.mp4"
        created_heat = [r for r in all_resources if r.name == name]
        if created_heat:
            pass
            # print(f"EXISTS {name} -> Nothing to do")
        else:
            video_path = result_dir_path / name
            print(f"CREATE {name}")
            images_dir = result_dir_path / "q-value-heat"
            image_prefix = f"{prefix}-{combi}"
            image_count = len(
                list(
                    [i for i in images_dir.iterdir() if i.name.startswith(image_prefix)]
                )
            )
            image_pattern = f"{image_prefix}*.png"
            frame_rate = 1 + (image_count // 30)
            cmd = [
                "ffmpeg",
                "-framerate",
                str(frame_rate),
                "-pattern_type",
                "glob",
                "-i",
                f"{image_pattern}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path.absolute()),
            ]
            print(f"calling '{' '.join(cmd)}'\ncwd: {images_dir}")
            ok, stdout = hlp.call1(cmd, work_path=images_dir)
            if not ok:
                print(f"ERROR calling {cmd}\n{stdout}")

    def create_simulation_video(all_resources, combi, prefix, result_dir_path):
        name_regex = f"{prefix}-{combi}-.*sumosim-video.mp4"
        is_created = len(
            list([r for r in all_resources if re.match(name_regex, r.name)])
        )
        if is_created:
            pass
            # print(f"EXISTS {name_regex} -> Nothing to do")
        else:
            simulation_prefix = f"{prefix}-{combi}"
            cmd = [
                "sumo",
                "video",
                "-p",
                simulation_prefix,
                "-o",
                str(result_dir_path.absolute()),
            ]
            print(f"CREATE simulation video '{' '.join(cmd)}'")
            ok, stdout = hlp.call1(cmd)
            if not ok:
                print(f"ERROR calling {cmd}\n{stdout}")

    for m in reports_data["methods"]:
        for t in m["trainings"]:
            prefix = t["prefix"]
            for p in result_dir_paths:
                create(p, prefix)


def report(result_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_video_names = AllVideoNames()
    report_path = (
        Path(__file__).parent.parent.parent.parent / "resources" / "report.yml"
    )
    result_dir_paths = [d for d in result_dir.iterdir() if d.is_dir()]
    with report_path.open() as f:
        reports_data = yaml.safe_load(f)
    # pprint(reports_data)
    create_final_resources(reports_data, result_dir_paths)
    create_report(reports_data, result_dir_paths, out_dir, all_video_names)
