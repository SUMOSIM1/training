import training.sgym.qlearn as _ql
import pytest


max_value = 10.0
min_eps = 0.0005
max_eps = 0.01

fetch_type_data = [
    ([5, 8, 6, 3], _ql.FetchType.LAZY_L_T2, [5, 8]),
    ([5, 8, 6, 3], _ql.FetchType.LAZY_L, [3, 5, 6, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L_T2, [3, 6]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L, [3, 5, 6, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L_T5, [3, 5, 6, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L_T10, [3, 5, 6, 8]),
    ([5, 8, 6, 3], _ql.FetchType.LAZY_M_T2, [5, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_M_T2, [3, 6]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_M, [3, 5, 6, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_M_T5, [3, 5, 6, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_M_T10, [3, 5, 6, 8]),
    ([5, 8, 6, 3], _ql.FetchType.LAZY_L_T2, [5, 8]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L_T2, [3, 6]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L, [3, 5, 6, 8]),
    ([3, 5, 8, 10, 12], _ql.FetchType.LAZY_L, [3, 5, 8, 10, 12]),
    ([6, 3, 5, 8], _ql.FetchType.LAZY_L_T5, [3, 5, 6, 8]),
    ([6, 3, 5, 8, 0, 1], _ql.FetchType.LAZY_L_T5, [0, 3, 5, 6, 8]),
    (
        [6, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        _ql.FetchType.LAZY_L_T10,
        [3, 5, 6, 8, 9, 10, 11, 12, 13, 14],
    ),
    (
        [6, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100],
        _ql.FetchType.LAZY_L_T10,
        [3, 5, 6, 8, 9, 10, 11, 12, 13, 14],
    ),
]


@pytest.mark.parametrize(
    "indexes_in_range, fetch_type, expected_indexes_proposed", fetch_type_data
)
def test_qlearn_fetch_type(
    indexes_in_range: list[int], fetch_type: _ql.FetchType, expected_indexes_proposed
):
    data = _create_test_data(indexes_in_range)
    indexes = set()
    for _ in range(100):
        indexes.add(fetch_type.fetch(data))
    _indexes = sorted(indexes)
    assert _indexes == expected_indexes_proposed


def _create_test_data(indexes_in_range: list[int]) -> list[float]:
    diff = min_eps / (len(indexes_in_range) + 1)

    def create_values_in_range() -> list[float]:
        _result = []
        curr = max_value
        for _ in range(len(indexes_in_range)):
            _result.append(curr)
            curr -= diff
        return _result

    in_values = create_values_in_range()
    assert in_values[-1] > max_value - min_eps
    out_value = max_value - max_eps - diff
    out_len = max(indexes_in_range) + 3
    result = [out_value] * out_len
    i_in = 0
    for i in indexes_in_range:
        result[i] = in_values[i_in]
        i_in += 1
    return result


create_test_data_data = [
    (
        [5, 8, 6, 3],
        [
            9.9899,
            9.9899,
            9.9899,
            9.9997,
            9.9899,
            10.0,
            9.9998,
            9.9899,
            9.9999,
            9.9899,
            9.9899,
        ],
    ),
    (
        [5, 8],
        [
            9.989833333333333,
            9.989833333333333,
            9.989833333333333,
            9.989833333333333,
            9.989833333333333,
            10.0,
            9.989833333333333,
            9.989833333333333,
            9.999833333333333,
            9.989833333333333,
            9.989833333333333,
        ],
    ),
    (
        [5],
        [9.98975, 9.98975, 9.98975, 9.98975, 9.98975, 10.0, 9.98975, 9.98975],
    ),
    (
        [6, 3, 5, 8],
        [
            9.9899,
            9.9899,
            9.9899,
            9.9999,
            9.9899,
            9.9998,
            10.0,
            9.9899,
            9.9997,
            9.9899,
            9.9899,
        ],
    ),
]


@pytest.mark.parametrize("indexes_in_range, expected_test_data", create_test_data_data)
def test_create_test_data(indexes_in_range: list[int], expected_test_data):
    test_data = _create_test_data(indexes_in_range)
    assert test_data == expected_test_data


epsilon_decay_data = [
    (0, 1.0, _ql.EpsilonDecay.NONE, 1.0),
    (0, 2.0, _ql.EpsilonDecay.NONE, 2.0),
    (10, 1.0, _ql.EpsilonDecay.NONE, 1.0),
    (20000, 1.0, _ql.EpsilonDecay.NONE, 1.0),
    (0, 2.0, _ql.EpsilonDecay.DECAY_100_50, 2.0),
    (100, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.0),
    (101, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.0),
    (1000, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.0),
    (50000, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.0),
    (25, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.75),
    (50, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.5),
    (75, 2.0, _ql.EpsilonDecay.DECAY_100_50, 1.25),
    (50, 1.0, _ql.EpsilonDecay.DECAY_100_80, 0.9),
    (100, 1.0, _ql.EpsilonDecay.DECAY_100_80, 0.8),
    (250, 1.0, _ql.EpsilonDecay.DECAY_100_80, 0.8),
    (50, 1.0, _ql.EpsilonDecay.DECAY_100_20, 0.6),
    (500, 1.0, _ql.EpsilonDecay.DECAY_1000_80, 0.9),
    (750, 2.0, _ql.EpsilonDecay.DECAY_1000_50, 1.25),
    (500, 1.0, _ql.EpsilonDecay.DECAY_1000_20, 0.6),
    (1500, 1.0, _ql.EpsilonDecay.DECAY_3000_80, 0.9),
    (2250, 2.0, _ql.EpsilonDecay.DECAY_3000_50, 1.25),
    (1500, 1.0, _ql.EpsilonDecay.DECAY_3000_20, 0.6),
]


@pytest.mark.parametrize(
    "epochs, initial_epsilon, epsilon_decay, expected_epsilon", epsilon_decay_data
)
def test_epsilon_decay(
    epochs: int,
    initial_epsilon: float,
    epsilon_decay: _ql.EpsilonDecay,
    expected_epsilon: float,
):
    epsilon = epsilon_decay.epsilon(epochs, initial_epsilon)
    assert epsilon == expected_epsilon
