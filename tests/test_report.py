import pytest
import training.explore.report as _r


format_combi_id_data = {
    ("A1", "A1"),
    ("AB1", "AB1"),
    ("AB10", "AB10"),
    ("B2", "B2"),
    ("B2D3", "B2 D3"),
    ("BX2D3", "BX2 D3"),
    ("B2DX3", "B2 DX3"),
    ("BX20D3", "BX20 D3"),
    ("BX20D333", "BX20 D333"),
    ("B2D3C4", "B2 D3 C4"),
    ("B2DX3C400", "B2 DX3 C400"),
}


@pytest.mark.parametrize("combi_id, expected", format_combi_id_data)
def test_format_combi_id(combi_id: str, expected: str):
    assert _r.format_combi_id(combi_id) == expected
