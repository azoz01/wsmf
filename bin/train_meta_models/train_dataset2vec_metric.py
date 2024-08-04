import pytorch_lightning as pl
import torch
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from loguru import logger
from pytorch_lightning import Trainer

from experiments_engine.data_utils import load_datasets_with_landmarkers
from experiments_engine.paths import paths_provider
from wsmf.metamodels.data import (
    EncoderHpoDataset,
    EncoderMetricLearningLoader,
    GenericRepeatableDataLoader,
)
from wsmf.metamodels.networks import Dataset2VecMetricLearning

torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")


def main():
    pl.seed_everything(123)
    logger.info("Preparing dataloaders")
    train_datasets, train_landmarkers, val_datasets, val_landmarkers = (
        load_datasets_with_landmarkers()
    )
    train_dataset = EncoderHpoDataset(train_datasets, train_landmarkers)
    train_dataloader = EncoderMetricLearningLoader(train_dataset, 128, 5)
    val_dataset = EncoderHpoDataset(val_datasets, val_landmarkers)
    val_dataloader = EncoderMetricLearningLoader(val_dataset, 128, 8)
    val_dataloader = GenericRepeatableDataLoader(val_dataloader)

    logger.info("Training meta-model")
    model = Dataset2VecMetricLearning(
        config=Dataset2VecConfig(
            f_res_n_layers=1,
            f_block_repetitions=1,
            f_out_size=512,
            f_dense_hidden_size=512,
            g_layers_sizes=[512] * 1,
            h_res_n_layers=1,
            h_block_repetitions=1,
            h_res_hidden_size=512,
            h_dense_hidden_size=512,
            output_size=512,
            activation_cls=torch.nn.GELU,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=1e-4, weight_decay=0, gamma=1
        ),
    ).cuda()
    trainer = Trainer(
        max_epochs=500,
        log_every_n_steps=1,
        default_root_dir=paths_provider.encoders_results_path / "d2v_metric",
        check_val_every_n_epoch=2,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    logger.info("Finished")


if __name__ == "__main__":
    main()
