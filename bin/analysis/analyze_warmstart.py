import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from experiments_engine.cd_plot import draw_cd_diagram
from experiments_engine.paths import paths_provider


def main():
    logger.info("Loading data")
    df = pd.read_csv(paths_provider.warmstart_results_path / "xgboost.csv")

    logger.info("Processing data")
    aggregates = (
        df.groupby(["dataset"], as_index=False)
        .agg({"value": ["min", "max"]})
        .reset_index(drop=True)
    )
    aggregates.columns = ["dataset", "min_value", "max_value"]

    plot_data = df.merge(aggregates, "left", "dataset")
    plot_data = plot_data.sort_values(["dataset", "warmstart", "number"])
    plot_data["cumulative_max_value"] = plot_data.groupby(
        ["dataset", "warmstart"]
    )["value"].cummax()

    plot_data["distance"] = (
        plot_data["max_value"] - plot_data["cumulative_max_value"]
    ) / (plot_data["max_value"] - plot_data["min_value"])
    plot_data["distance"] = (
        plot_data["max_value"] - plot_data["cumulative_max_value"]
    ) / (plot_data["max_value"] - plot_data["min_value"])
    plot_data["scaled_value"] = (df["value"] - plot_data["min_value"]) / (
        plot_data["max_value"] - plot_data["min_value"]
    )

    logger.info("Generating plot for raw values")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(
        plot_data, x="number", y="scaled_value", hue="warmstart", ax=ax
    )
    plt.savefig(paths_provider.results_analysis_path / "raw_values.png")
    plt.clf()

    logger.info("Generating ADTM plot")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(data=plot_data, x="number", y="distance", hue="warmstart")
    plt.savefig(paths_provider.results_analysis_path / "adtm.png")
    plt.clf()

    logger.info("Generating CD plot")
    plot_data["accuracy"] = 1 - plot_data["distance"]
    plot_data["classifier_name"] = plot_data["warmstart"]
    plot_data["dataset_name"] = plot_data["dataset"]
    draw_cd_diagram(
        df_perf=plot_data[["classifier_name", "dataset_name", "accuracy"]].loc[
            plot_data.number == 4
        ]
    )
    plt.savefig(
        paths_provider.results_analysis_path / "cd.png", bbox_inches="tight"
    )
    plt.clf()


if __name__ == "__main__":
    main()
