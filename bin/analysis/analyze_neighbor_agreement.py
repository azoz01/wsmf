import json
from functools import partial
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
from dataset2vec.model import Dataset2Vec
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from experiments_engine.data_utils import load_datasets_with_landmarkers
from experiments_engine.paths import paths_provider
from experiments_engine.selectors_utils import get_hp_selector_from_path
from wsmf.metamodels.networks import (
    Dataset2VecForLandmarkerReconstruction,
    Dataset2VecMetricLearning,
)
from wsmf.selectors.baselines import (
    LandmarkerHpSelector,
    RandomHpSelector,
    RankBasedHpSelector,
)
from wsmf.selectors.reconstruction_based import ReconstructionBasedHpSelector
from wsmf.selectors.representation_based import RepresentationBasedHpSelector
from wsmf.selectors.selector import WarmstartHpSelector


def calculate_neighbor_agreement(
    metadataset: dict[str, Tuple[Tensor, Tensor]],
    landmarkers: dict[str, Tensor],
    selector: WarmstartHpSelector,
    landmarker_hp_selector: LandmarkerHpSelector,
    top_n: int,
) -> Tuple[float, float]:
    common_configurations = []
    for dataset_name, dataset in tqdm(metadataset.items()):
        proposed_configuration_idx = selector.propose_configurations_idx(
            dataset, top_n
        )
        landmarker_configuration_idx = (
            landmarker_hp_selector.propose_configurations_idx(
                landmarkers[dataset_name], top_n
            )
        )
        n_common = len(
            set(proposed_configuration_idx).intersection(
                landmarker_configuration_idx
            )
        )
        common_configurations.append(n_common)
    return (
        np.mean(common_configurations).item(),
        np.std(common_configurations).item(),
    )


def main():
    pl.seed_everything(123)
    logger.info("Loading data")
    _, _, val_datasets, val_landmarkers = load_datasets_with_landmarkers()

    logger.info("Loading encoders")
    selectors = [
        (
            "Random from portfolio",
            get_hp_selector_from_path(
                RandomHpSelector,
                paths_provider.train_meta_dataset_path,
                paths_provider.hp_portfolio_configurations_path
                / "xgboost.json",
                paths_provider.landmarkers_path / "xgboost.json",
            ),
        ),
        (
            "Rank from portfolio",
            get_hp_selector_from_path(
                RankBasedHpSelector,
                paths_provider.train_meta_dataset_path,
                paths_provider.hp_portfolio_configurations_path
                / "xgboost.json",
                paths_provider.landmarkers_path / "xgboost.json",
            ),
        ),
        (
            "Dataset2Vec basic",
            get_hp_selector_from_path(
                partial(
                    RepresentationBasedHpSelector,
                    encoder=Dataset2Vec.load_from_checkpoint(
                        list(
                            (
                                paths_provider.encoders_results_path
                                / "d2v_base"
                            ).rglob("*.ckpt")
                        )[0]
                    ),
                ),
                paths_provider.train_meta_dataset_path,
                paths_provider.hp_portfolio_configurations_path
                / "xgboost.json",
                paths_provider.landmarkers_path / "xgboost.json",
            ),
        ),
        (
            "Dataset2Vec metric learning",
            get_hp_selector_from_path(
                partial(
                    RepresentationBasedHpSelector,
                    encoder=Dataset2VecMetricLearning.load_from_checkpoint(
                        list(
                            (
                                paths_provider.encoders_results_path
                                / "d2v_metric"
                            ).rglob("*.ckpt")
                        )[0]
                    ),
                ),
                paths_provider.train_meta_dataset_path,
                paths_provider.hp_portfolio_configurations_path
                / "xgboost.json",
                paths_provider.landmarkers_path / "xgboost.json",
            ),
        ),
        (
            "Dataset2Vec landmarker reconstruction (landmarkers)",
            get_hp_selector_from_path(
                partial(
                    ReconstructionBasedHpSelector,
                    encoder=Dataset2VecForLandmarkerReconstruction.load_from_checkpoint(  # noqa: E501
                        list(
                            (
                                paths_provider.encoders_results_path
                                / "d2v_reconstruction"
                            ).rglob("*.ckpt")
                        )[0]
                    ),
                ),
                paths_provider.train_meta_dataset_path,
                paths_provider.hp_portfolio_configurations_path
                / "xgboost.json",
                paths_provider.landmarkers_path / "xgboost.json",
            ),
        ),
        (
            "Dataset2Vec reconstruction (representations)",
            get_hp_selector_from_path(
                partial(
                    RepresentationBasedHpSelector,
                    encoder=Dataset2VecForLandmarkerReconstruction.load_from_checkpoint(  # noqa: E501
                        list(
                            (
                                paths_provider.encoders_results_path
                                / "d2v_reconstruction"
                            ).rglob("*.ckpt")
                        )[0]
                    ),
                ),
                paths_provider.train_meta_dataset_path,
                paths_provider.hp_portfolio_configurations_path
                / "xgboost.json",
                paths_provider.landmarkers_path / "xgboost.json",
            ),
        ),
    ]

    landmarker_hp_selector = get_hp_selector_from_path(
        LandmarkerHpSelector,
        paths_provider.train_meta_dataset_path,
        paths_provider.hp_portfolio_configurations_path / "xgboost.json",
        paths_provider.landmarkers_path / "xgboost.json",
    )

    logger.info("Analyzing")
    results = dict()
    for name, selector in selectors:
        logger.info(f"Analyzing encoder - {name}")
        ranks_mean, ranks_std = calculate_neighbor_agreement(
            val_datasets, val_landmarkers, selector, landmarker_hp_selector, 5
        )
        results[name] = {
            "mean": ranks_mean,
            "std": ranks_std,
        }

    with open(
        paths_provider.results_analysis_path / "neighbor_agreement.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
