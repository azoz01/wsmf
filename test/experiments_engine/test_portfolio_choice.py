import numpy as np

from experiments_engine.portfolio_selection import (
    extract_best_configuration_idx_from_cluster_eval_results,
)


def test_extract_best_configuration_idx_from_cluster_eval_results() -> None:
    # Given
    datasets_inside_clusters_performances = np.array(
        [
            [0.5, 0.6, 0.5, 0.7],
            [0.5, 0.6, 0.7, 0.6],
            [0.5, 0.7, 0.5, 0.7],
            [0.4, 0.6, 0.5, 0.4],
        ]
    )

    # When
    actual_idx = extract_best_configuration_idx_from_cluster_eval_results(
        datasets_inside_clusters_performances
    )

    # Then
    assert actual_idx == 1
