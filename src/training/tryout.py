import training.helper as hlp
from pathlib import Path


def main():
    md_txt = """
# This i a table tryout

some normal text

&#x200B;      | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell

""".strip()

    html = hlp.parse_markdown(md_txt)

    out = Path.home() / "tmp" / "sumosim" / "mdtryout.html"

    with out.open("w") as f:
        f.write(html)

    print(f"wrote to: {out}")
