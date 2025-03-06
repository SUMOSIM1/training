import pytest
import training.sgym.qtables as qt


from_key_data = [
    ((0, 1, 2, 3), "0|1|2|3"),
    ((0, 1, 2, -5), "0|1|2|-5"),
    ((0, 0, 2, 3), "0|0|2|3"),
    ((0, 1, 2, 1000), "0|1|2|1000"),
]


@pytest.mark.parametrize("key, expected", from_key_data)
def test_from_key(key: tuple[int, int, int, int], expected: str):
    v = qt.from_key(key)
    assert v == expected


to_key_data = [
    ("0|1|2|3", (0, 1, 2, 3)),
    ("0|1|2|-5", (0, 1, 2, -5)),
    ("0|0|2|3", (0, 0, 2, 3)),
    ("0|1|2|1000", (0, 1, 2, 1000)),
]


@pytest.mark.parametrize("strval, expected", to_key_data)
def test_to_key(strval: str, expected: tuple[int, int, int, int]):
    k = qt.to_key(strval)
    assert k == expected
