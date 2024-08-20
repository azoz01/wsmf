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
    plot_data = plot_data.sort_values(
        ["dataset", "warmstart", "number", "value"]
    )
    plot_data["cumulative_max_value"] = plot_data.groupby(
        ["dataset", "warmstart"]
    )["value"].cummax()
    plot_data["neg_value"] = -plot_data["value"]
    plot_data = plot_data.sort_values(
        ["dataset", "warmstart", "number", "neg_value"]
    )
    plot_data["rank"] = plot_data.groupby(["dataset", "number"])[
        "neg_value"
    ].rank()

    plot_data["distance"] = (
        plot_data["max_value"] - plot_data["cumulative_max_value"]
    ) / (plot_data["max_value"] - plot_data["min_value"])
    plot_data["distance"] = (
        plot_data["max_value"] - plot_data["cumulative_max_value"]
    ) / (plot_data["max_value"] - plot_data["min_value"])
    plot_data["scaled_value"] = (df["value"] - plot_data["min_value"]) / (
        plot_data["max_value"] - plot_data["min_value"]
    )
    plot_data["number"] = plot_data["number"] + 1

    logger.info("Generating plot of ranks over time")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(plot_data, x="number", y="rank", hue="warmstart", ax=ax)
    ax.vlines(
        x=5,
        ymin=1,
        ymax=7,
        colors="black",
        label="End of warm-start phase",
        linestyles="dotted",
    )
    ax.set_ylabel("Rank")
    ax.set_xlabel("Number of iteration")
    ax.set_title("Rank of the negative ROC AUC over time")
    plt.savefig(paths_provider.results_analysis_path / "rank_over_time.png")
    plt.clf()

    logger.info("Generating plot for raw values")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        plot_data, x="number", y="scaled_value", hue="warmstart", ax=ax
    )
    ax.vlines(
        x=5,
        ymin=0,
        ymax=1,
        colors="black",
        label="End of warm-start phase",
        linestyles="dotted",
    )
    ax.set_ylabel("Scaled ROC AUC")
    ax.set_xlabel("Number of iteration")
    ax.set_title("Scaled value of the ROC AUC over time")
    plt.savefig(paths_provider.results_analysis_path / "raw_values.png")
    plt.clf()

    logger.info("Generating ADTM plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=plot_data, x="number", y="distance", hue="warmstart", ax=ax
    )
    ax.vlines(
        x=5,
        ymin=0,
        ymax=0.25,
        colors="black",
        label="End of warm-start phase",
        linestyles="dotted",
    )
    ax.set_xlabel("Number of iteration")
    ax.set_ylabel("Scaled distance")
    plt.savefig(paths_provider.results_analysis_path / "adtm.png")
    plt.clf()

    logger.info("Generating CD plot")
    plot_data["accuracy"] = 1 - plot_data["distance"]
    plot_data["classifier_name"] = plot_data["warmstart"]
    plot_data["dataset_name"] = plot_data["dataset"]
    draw_cd_diagram(
        df_perf=plot_data[["classifier_name", "dataset_name", "accuracy"]].loc[
            plot_data.number == 5
        ]
    )
    plt.savefig(
        paths_provider.results_analysis_path / "cd_5.png", bbox_inches="tight"
    )
    plt.clf()

    draw_cd_diagram(
        df_perf=plot_data[["classifier_name", "dataset_name", "accuracy"]].loc[
            plot_data.number == 20
        ]
    )
    plt.savefig(
        paths_provider.results_analysis_path / "cd_20.png", bbox_inches="tight"
    )
    plt.clf()


if __name__ == "__main__":
    main()
