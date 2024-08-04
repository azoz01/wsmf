import json
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import ttest_1samp
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from experiments_engine.paths import paths_provider


def analyze_optimization(
    df: pd.DataFrame, param_columns: list[str]
) -> dict[str, float]:
    early_iterations = df[df.number <= 4]
    optim_iterations = df[df.number > 4]
    distance_within_early = np.mean(
        [
            np.linalg.norm(row1 - row2)
            for row1, row2 in combinations(
                early_iterations[param_columns].values, 2
            )
        ]
    )
    distance_within_optim = np.mean(
        [
            np.linalg.norm(row1 - row2)
            for row1, row2 in combinations(
                optim_iterations[param_columns].values, 2
            )
        ]
    )
    distance_between = np.mean(
        [
            np.linalg.norm(row1 - row2)
            for row1, row2 in product(
                optim_iterations[param_columns].values,
                early_iterations[param_columns].values,
            )
        ]
    )
    return {
        "distance_within_early": distance_within_early,
        "distance_within_optim": distance_within_optim,
        "distance_between": distance_between,
    }


def main():
    logger.info("Loading data")
    df = pd.read_csv(paths_provider.warmstart_results_path / "xgboost.csv")
    df = df[df.warmstart != "No warmstart"]

    logger.info("Processing data")
    param_columns = list(
        filter(lambda col: col.startswith("params"), df.columns)
    )
    columns_to_analyze = param_columns + ["number", "dataset", "warmstart"]
    logarithmic_columns = [
        "params_eta",
        "params_gamma",
        "params_min_child_weight",
        "params_reg_lambda",
        "params_reg_alpha",
    ]
    df_params = df[columns_to_analyze].copy()
    df_params[logarithmic_columns] = np.log(df_params[logarithmic_columns])
    df_params[param_columns] = (
        MinMaxScaler()
        .set_output(transform="pandas")
        .fit_transform(df_params[param_columns])
    )

    logger.info("Analyzing data")
    results = []
    for dataset, warmstart in tqdm(
        df_params[["dataset", "warmstart"]].drop_duplicates().values
    ):
        df_optimization = df_params.loc[
            (df_params.dataset == dataset) & (df_params.warmstart == warmstart)
        ]
        results.append(analyze_optimization(df_optimization, param_columns))
    distance_analysis_results = pd.DataFrame(results)

    analysis_result = dict()
    logger.info("Analyzing distance_within_optim/distance_within_early")
    analysis_col = (
        distance_analysis_results.distance_within_optim
        / distance_analysis_results.distance_within_early
    )
    test_result = ttest_1samp(analysis_col, 1, alternative="greater")
    analysis_result["distance_within_optim/distance_within_early"] = {
        "pvalue": test_result.pvalue,
        "mean": analysis_col.mean(),
        "std": analysis_col.std(),
    }

    logger.info("Analyzing distance_between/distance_within_early")
    analysis_col = (
        distance_analysis_results.distance_between
        / distance_analysis_results.distance_within_early
    )
    test_result = ttest_1samp(analysis_col, 1, alternative="greater")
    analysis_result["distance_between/distance_within_early"] = {
        "pvalue": test_result.pvalue,
        "mean": analysis_col.mean(),
        "std": analysis_col.std(),
    }
    with open(
        paths_provider.meta_dataset_analysis_path / "hp_distances.json", "w"
    ) as f:
        json.dump(analysis_result, f, indent=4)

    logger.info("Generating hyperparameter scatter plot")

    def binarize_number(number: int) -> str:
        if number <= 4:
            return "warm-start"
        else:
            return "optimization"

    optimizations = (
        df_params[["dataset", "warmstart"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sample(n=50)
        .values
    )
    df_params_sampled = pd.concat(
        [
            df_params.loc[
                (df_params.dataset == dataset)
                & (df_params.warmstart == warmstart)
            ]
            for dataset, warmstart in optimizations
        ]
    )

    df_params_sampled[["projected_x", "projected_y"]] = MDS(
        n_components=2
    ).fit_transform(df_params_sampled[param_columns])
    df_params_sampled["Optimization phase"] = df_params_sampled[
        "number"
    ].apply(binarize_number)
    sns.scatterplot(
        data=df_params_sampled,
        x="projected_x",
        y="projected_y",
        hue="Optimization phase",
    )
    plt.savefig(paths_provider.meta_dataset_analysis_path / "hp_scatter.png")


if __name__ == "__main__":
    main()
