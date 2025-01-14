from pathlib import Path
import shutil


def main():
    print("-- rename")
    indir = Path.home() / "tmp" / "sumosim" / "video"
    outdir = Path.home() / "tmp" / "sumosim" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    for f in indir.iterdir():
        if f.name.startswith("sumosim-video"):
            stem = f.stem.split("-")
            new_name = f"{stem[2]}-{stem[3]}-{stem[4]}-sumosim-video.mp4"
            target = outdir / new_name
            if not target.exists():
                shutil.copy(f, target)
                print(f"copied to {target}")
        elif f.stem.endswith("boxplot"):
            target = outdir / f.name
            if not target.exists():
                shutil.copy(f, target)
                print(f"copied to {target}")
        else:
            stem = f.stem.split("-")
            if len(stem) == 2 and f.suffix == ".mp4":
                new_name = f"{stem[0]}-{stem[1]}-q-values-heat.mp4"
                target = outdir / new_name
                if not target.exists():
                    shutil.copy(f, target)
                    print(f"copied to {target}")
