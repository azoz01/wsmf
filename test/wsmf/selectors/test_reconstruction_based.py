from unittest.mock import Mock

from torch import Tensor, rand

from wsmf.selectors import ReconstructionBasedHpSelector


def test_reconstruction_based_hp_selector() -> None:
    # Given
    selector = ReconstructionBasedHpSelector(
        Mock(return_value=Tensor([0.35, 0.35, 0.35])),
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
        (rand((5, 3)), rand((5, 1))), 2
    )

    # Then
    assert proposed_configurations == [
        {"hparam2": 2},
        {"hparam3": 3},
    ]
