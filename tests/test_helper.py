import pytest
import training.helper as hlp
import math


@pytest.mark.parametrize(
    "start_value, half_time, t, expected_result",
    [
        (0.1, 100, 0, 0.1),
        (0.1, 100, 50, 0.07071067811865477),
        (0.1, 100, 100, 0.05),
        (0.1, 100, 200, 0.025),
        (0.1, 100, 500, 0.003125),
        (0.1, 100, 1000, 9.765625e-05),
        (0.01, 100, 0, 0.01),
        (0.01, 100, 50, 0.007071067811865477),
        (0.01, 100, 100, 0.005),
        (0.01, 100, 200, 0.0025),
        (0.01, 100, 500, 0.0003125),
        (0.01, 100, 1000, 9.765625e-06),
    ],
)
def test_descending_exponential_valid_inputs(
    start_value, half_time, t, expected_result
):
    actual_result = hlp.descending_exponential(start_value, half_time, t)
    assert math.isclose(actual_result, expected_result, rel_tol=1e-9, abs_tol=1e-12)
