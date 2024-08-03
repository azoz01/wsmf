from unittest.mock import Mock, patch

import numpy as np
from torch import Tensor, rand

from wsmf.selectors import (
    LandmarkerHpSelector,
    RandomHpSelector,
    RankBasedHpSelector,
)


@patch("numpy.random.choice")
def test_random_hp_selector(choice_mock: Mock) -> None:
    # Given
    choice_mock.return_value = np.array([0, 2])
    selector = RandomHpSelector(
        {
            "dataset1": (rand((5, 3)), rand((5, 1))),
            "dataset2": (rand((10, 2)), rand((10, 1))),
        },
        {
            "dataset1": Tensor([1, 2, 3]),
            "dataset2": Tensor([2, 3, 4]),
        },
        [
            {"hparam1": 1},
            {"hparam2": 2},
            {"hparam3": 3},
        ],
    )

    # When
    proposed_configurations = selector.propose_configurations(
        (rand((4, 3)), rand((4, 1))), 2
    )

    # Then
    assert proposed_configurations == [
        {"hparam1": 1},
        {"hparam3": 3},
    ]


def test_rank_based_hp_selector() -> None:
    # Given
    selector = RankBasedHpSelector(
        {
            "dataset1": (rand((5, 3)), rand((5, 1))),
            "dataset2": (rand((10, 2)), rand((10, 1))),
            "dataset3": (rand((10, 2)), rand((10, 1))),
        },
        {
            "dataset1": Tensor([1, 2, 3]),
            "dataset2": Tensor([2, 3, 2]),
            "dataset3": Tensor([3, 4, 1]),
        },
        [
            {"hparam1": 1},
            {"hparam2": 2},
            {"hparam3": 3},
        ],
    )

    # When
    proposed_configurations = selector.propose_configurations(
        (rand((4, 3)), rand((4, 1))), 2
    )

    # Then
    assert proposed_configurations == [
        {"hparam2": 2},
        {"hparam3": 3},
    ]


def test_landmarker_hp_selector() -> None:
    # Given
    selector = LandmarkerHpSelector(
        {
            "dataset1": (rand((5, 3)), rand((5, 1))),
            "dataset2": (rand((10, 2)), rand((10, 1))),
            "dataset3": (rand((10, 2)), rand((10, 1))),
        },
        {
            "dataset1": Tensor([0.03, 0.01, 0.01]),
            "dataset2": Tensor([0.2, 0.3, 0.2]),
            "dataset3": Tensor([0.4, 0.5, 0.6]),
        },
        [
            {"hparam1": 1},
            {"hparam2": 2},
            {"hparam3": 3},
        ],
    )

    # When
    proposed_configurations = selector.propose_configurations(
        Tensor([0.35, 0.35, 0.35]), 2  # type: ignore
    )

    # Then
    assert proposed_configurations == [
        {"hparam2": 2},
        {"hparam3": 3},
    ]
