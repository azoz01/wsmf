import json

import numpy as np
import pytorch_lightning as pl
from dataset2vec.model import Dataset2Vec
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

from experiments_engine.data_utils import load_datasets_with_landmarkers
from experiments_engine.paths import paths_provider
from wsmf.metamodels.networks import (
    Dataset2VecForLandmarkerReconstruction,
    Dataset2VecMetricLearning,
)


def calculate_distance(v1, v2):
    return ((v1 - v2) ** 2).mean().item()


def main():
    pl.seed_everything(123)
    logger.info("Loading data")
    _, _, val_datasets, val_landmarkers = load_datasets_with_landmarkers()
    datasets_names = list(val_datasets.keys())

    logger.info("Loading encoders")
    considered_encoders = [
        (
            "Dataset2Vec metric learning",
            Dataset2VecMetricLearning.load_from_checkpoint(
                list(
                    (
                        paths_provider.encoders_results_path / "d2v_metric"
                    ).rglob("*.ckpt")
                )[0]
            ),
        ),
        (
            "Dataset2Vec basic",
            Dataset2Vec.load_from_checkpoint(
                list(
                    (paths_provider.encoders_results_path / "d2v_base").rglob(
                        "*.ckpt"
                    )
                )[0]
            ),
        ),
    ]
    logger.info("Analyzing")
    results = dict()
    for name, encoder in considered_encoders:
        logger.info(f"Analyzing encoder - {name}")
        correlations = []
        for _ in range(20):
            landmarkers_distances = []
            datasets_distances = []
            for _ in tqdm(range(1000)):
                d1_idx, d2_idx = np.random.choice(
                    len(datasets_names), 2, replace=False
                )
                name1 = datasets_names[d1_idx]
                name2 = datasets_names[d2_idx]
                l1, l2 = val_landmarkers[name1], val_landmarkers[name2]
                enc, enc2 = encoder(*val_datasets[name1]), encoder(
                    *val_datasets[name2]
                )
                landmarkers_distances.append(calculate_distance(l1, l2))
                datasets_distances.append(calculate_distance(enc, enc2))
            correlations.append(
                spearmanr(landmarkers_distances, datasets_distances).statistic
            )
        results[name] = {
            "mean": np.mean(correlations),
            "std": np.std(correlations),
        }
    logger.info("Analyzing encoder Dataset2Vec reconstruction")
    reconstruction_encoder = (
        Dataset2VecForLandmarkerReconstruction.load_from_checkpoint(
            list(
                (
                    paths_provider.encoders_results_path / "d2v_reconstruction"
                ).rglob("*.ckpt")
            )[0]
        )
    )
    correlations = []
    for _ in range(20):
        landmarkers_distances = []
        datasets_distances = []
        for _ in tqdm(range(1000)):
            d1_idx, d2_idx = np.random.choice(
                len(datasets_names), 2, replace=False
            )
            name1 = datasets_names[d1_idx]
            name2 = datasets_names[d2_idx]
            l1, l2 = val_landmarkers[name1], val_landmarkers[name2]
            enc = reconstruction_encoder(*val_datasets[name1])
            landmarkers_distances.append(calculate_distance(l1, l2))
            datasets_distances.append(calculate_distance(enc, l2))
        correlations.append(
            spearmanr(landmarkers_distances, datasets_distances).statistic
        )
    results["Dataset2Vec reconstruction"] = {
        "mean": np.mean(correlations),
        "std": np.std(correlations),
    }

    with open(
        paths_provider.results_analysis_path / "distances_correlation.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
