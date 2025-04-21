import collections
from pathlib import Path


class AllVideoNames:
    def __init__(self):
        self.res_dict = collections.defaultdict(list)
        for k, v in self.read_all_videos_file():
            self.res_dict[k].append(v)

    def video_names(self, key: str) -> list[str] | None:
        return self.res_dict.get(key)

    def split_line(self, line: str) -> tuple[str, str] | None:
        try:
            aa = line.split("/")
            res = aa[3]
            path = f"{aa[2]}/{res}"
            if res.endswith("\n"):
                res = res[:-1]
            cs = res.split("-")
            key = AllVideoNames.create_key(cs[0], cs[1])
            return key, path
        except BaseException as e:
            print(f"Error handling '{line}'")
            print(f"Error {e}")
            return None

    def read_all_videos_file(self) -> list[str]:
        res_dir = Path(__file__).parent.parent.parent.parent / "resources"
        file = res_dir / "all-videos.txt"
        lines = []
        cnt = 0
        with file.open() as f:
            line = f.readline()
            while line:
                kv = self.split_line(line)
                lines.append(kv)
                cnt += 1
                line = f.readline()
        return lines

    @staticmethod
    def create_key(prefix: str, combi: str) -> str:
        return f"{prefix}-{combi}"
