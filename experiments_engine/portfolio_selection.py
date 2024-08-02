import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


def extract_best_configuration_idx_from_cluster_eval_results(
    datasets_inside_clusters_performances: NDArray,
) -> int:
    return get_ranks_of_hp_configurations(
        datasets_inside_clusters_performances
    )[0]


def get_ranks_of_hp_configurations(hp_performances: NDArray) -> list[int]:
    ranks_per_dataset = np.array(
        [rankdata(-row, method="dense") for row in hp_performances]
    )
    average_ranks_per_configuration = ranks_per_dataset.mean(axis=0)
    final_ranks = np.argsort(average_ranks_per_configuration)
    return final_ranks.tolist()
