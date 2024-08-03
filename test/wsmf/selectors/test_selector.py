from typing import Tuple

from torch import Tensor, rand

from wsmf.selectors.selector import WarmstartHpSelector


class MockSelector(WarmstartHpSelector):

    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return [2, 0, 1]


def test_propose_configurations() -> None:
    # Given
    selector = MockSelector(
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
        (rand((5, 3)), rand((5, 1))), 3
    )

    # Then
    assert proposed_configurations == [
        {"hparam3": 3},
        {"hparam1": 1},
        {"hparam2": 2},
    ]
