import warnings

import pytorch_lightning as pl
import torch
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from experiments_engine.metamodels.dataset import (
    D2vHpClassificationDataset,
    GenericD2vHpoDataLoaderForHpo,
)
from experiments_engine.metamodels.networks.d2v_classification import (
    Dataset2VecTopKClassifier,
)
from experiments_engine.metamodels.scripts_utils import (
    load_datasets_with_landmarkers,
)
from experiments_engine.paths import paths_provider

warnings.simplefilter("ignore")
torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")


def main():
    pl.seed_everything(123)
    logger.info("Preparing dataloaders")
    train_datasets, train_landmarkers, val_datasets, val_landmarkers = (
        load_datasets_with_landmarkers()
    )
    train_dataset = D2vHpClassificationDataset(
        train_datasets, train_landmarkers, 5
    )
    train_dataloader = GenericD2vHpoDataLoaderForHpo(train_dataset, 5, True)
    val_dataset = D2vHpClassificationDataset(val_datasets, val_landmarkers, 5)
    val_dataloader = GenericD2vHpoDataLoaderForHpo(val_dataset, 32, False)

    logger.info("Training meta-model")
    outptut_dim = len(list(train_landmarkers.values())[0])
    model = Dataset2VecTopKClassifier(
        landmarker_size=outptut_dim,
        config=Dataset2VecConfig(
            f_res_n_layers=1,
            f_block_repetitions=1,
            f_res_hidden_size=128,
            f_out_size=128,
            f_dense_hidden_size=128,
            g_layers_sizes=[128],
            h_res_n_layers=1,
            h_block_repetitions=1,
            h_res_hidden_size=64,
            h_dense_hidden_size=64,
            output_size=64,
            activation_cls=torch.nn.GELU,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=1e-4, weight_decay=0, gamma=5
        ),
    ).cuda()
    trainer = Trainer(
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                save_top_k=1,
                mode="min",
                every_n_epochs=1,
                monitor="val_loss",
            ),
        ],
        log_every_n_steps=1,
        default_root_dir=paths_provider.encoders_results_path
        / "d2v_top_k_classification",
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    logger.info("Finished")


if __name__ == "__main__":
    main()
