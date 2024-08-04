import warnings
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from dataset2vec import (
    Dataset2Vec,
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from experiments_engine.paths import paths_provider

warnings.simplefilter("ignore")
torch.set_float32_matmul_precision("medium")


def main() -> None:
    pl.seed_everything(123)
    logger.info("Preparing data")
    train_loader = Dataset2VecLoader(
        [
            pd.read_parquet(path)
            for path in sorted(
                Path(
                    paths_provider.train_meta_dataset_path_for_plain_d2v
                ).iterdir()
            )
            if pd.read_parquet(path).shape[0] <= 5000
        ],
        batch_size=8,
        n_batches=64,
    )
    val_loader = RepeatableDataset2VecLoader(
        [
            pd.read_parquet(path)
            for path in sorted(
                Path(
                    paths_provider.val_meta_dataset_path_for_plain_d2v
                ).iterdir()
            )
            if pd.read_parquet(path).shape[0] <= 5000
        ],
        n_batches=32,
    )

    logger.info("Preparing model")
    model = Dataset2Vec(
        config=Dataset2VecConfig(
            f_res_n_layers=4,
            f_block_repetitions=1,
            f_res_hidden_size=512,
            f_out_size=512,
            f_dense_hidden_size=512,
            g_layers_sizes=[512],
            h_res_n_layers=4,
            h_block_repetitions=1,
            h_res_hidden_size=512,
            h_dense_hidden_size=512,
            output_size=256,
            activation_cls=torch.nn.GELU,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=1e-4, weight_decay=0, gamma=5
        ),
    )

    logger.info("Training model")
    trainer = Trainer(
        max_epochs=100_000,
        log_every_n_steps=5,
        default_root_dir=paths_provider.encoders_results_path / "d2v_base",
        callbacks=[
            EarlyStopping("val_accuracy", mode="max", patience=50),
            ModelCheckpoint(
                filename="{epoch}-{val_accuracy:.2f}-{train_accuracy:.2f}",
                save_top_k=1,
                mode="max",
                every_n_epochs=1,
                monitor="val_accuracy",
            ),
        ],
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
