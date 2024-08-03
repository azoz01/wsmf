import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from dataset2vec.utils import DataUtils
from loguru import logger
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from experiments_engine.paths import paths_provider

warnings.simplefilter("ignore")
pl.seed_everything(123)


def calculate_meta_features_record_from_path(
    dataset_path: Path,
) -> pd.DataFrame:
    dataset_name = dataset_path.name
    df = pd.read_parquet(dataset_path / "test.parquet")
    X, y = df.iloc[:, :-1], df.values[:, -1]
    X = DataUtils.get_preprocessing_pipeline().fit_transform(X).values  # type: ignore # noqa
    mfe = MFE(
        groups=[
            "general",
            "info-theory",
            "statistical",
        ],
        summary=["mean"],
    )
    mfe.fit(
        X,
        y,
        cat_cols=[],
        suppress_warnings=True,
        verbose=0,
    )
    ft = mfe.extract(out_type=pd.DataFrame)
    ft["dataset_name"] = dataset_name  # type: ignore
    return ft  # type: ignore


def main() -> None:
    logger.info("Calculating meta-features")
    paths = list(sorted(paths_provider.datasets_splitted_path.iterdir()))
    with Pool(14) as p:
        meta_features = list(
            tqdm(
                p.imap(calculate_meta_features_record_from_path, paths),
                total=len(paths),
            )
        )

    logger.info("Processing meta-features")
    meta_features_df = pd.concat(meta_features)
    meta_features_df = meta_features_df.drop(
        columns=[
            "num_to_cat",
            "sd_ratio",
            "lh_trace",
            "roy_root",
            "can_cor.mean",
            "nr_disc",
            "p_trace",
            "w_lambda",
        ]
    )
    meta_features_df.iloc[:, :-1] = meta_features_df.iloc[:, :-1].fillna(
        meta_features_df.iloc[:, :-1].mean(axis=0)
    )
    meta_features_df.iloc[:, :-1] = MinMaxScaler().fit_transform(
        meta_features_df.iloc[:, :-1]
    )

    logger.info("Saving meta-features")
    meta_features_df.to_parquet(paths_provider.metafeatures_path, index=False)


if __name__ == "__main__":
    main()
