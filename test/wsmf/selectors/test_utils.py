import numpy as np

from wsmf.selectors.utils import get_ranks_of_hp_configurations


def test_get_ranks_of_hp_configurations() -> None:
    # Given
    performance_matrix = np.array(
        [
            [0.5, 0.6, 0.5, 0.7],
            [0.5, 0.6, 0.7, 0.6],
            [0.5, 0.7, 0.5, 0.7],
            [0.4, 0.6, 0.5, 0.4],
        ]
    )

    # When
    actual_idx = get_ranks_of_hp_configurations(performance_matrix)

    # Then
    assert actual_idx == [1, 3, 2, 0]
