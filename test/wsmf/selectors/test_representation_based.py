from unittest.mock import Mock

from torch import Tensor, rand

from wsmf.selectors import RepresentationBasedHpSelector


def test_representation_based_hp_selector() -> None:
    # Given
    mock_representations_iterator = iter(
        [
            Tensor([0.03, 0.01, 0.01]),
            Tensor([0.2, 0.3, 0.2]),
            Tensor([0.2, 0.3, 0.2]),
            Tensor([0.35, 0.35, 0.35]),
        ]
    )
    selector = RepresentationBasedHpSelector(
        Mock(side_effect=lambda x, y: next(mock_representations_iterator)),
        {
            "dataset1": (rand((5, 3)), rand((5, 1))),
            "dataset2": (rand((10, 2)), rand((10, 1))),
            "dataset3": (rand((10, 2)), rand((10, 1))),
        },
        {
            "dataset1": Tensor([11, 2, 3]),
            "dataset2": Tensor([3, 10, 12]),
            "dataset3": Tensor([6, 9, 8]),
        },
        [
            {"hparam1": 1},
            {"hparam2": 2},
            {"hparam3": 3},
        ],
    )

    # When
    proposed_configurations = selector.propose_configurations(
        (rand((5, 3)), rand((5, 1))), 2
    )

    # Then
    assert proposed_configurations == [
        {"hparam3": 3},
        {"hparam2": 2},
    ]
