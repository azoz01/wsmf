from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


def get_ranks_of_hp_configurations(hp_performances: NDArray[Any]) -> list[int]:
    ranks_per_dataset = np.array(
        [rankdata(-row, method="dense") for row in hp_performances]
    )
    average_ranks_per_configuration = ranks_per_dataset.mean(axis=0)
    final_ranks = np.argsort(average_ranks_per_configuration)
    return list(final_ranks)
