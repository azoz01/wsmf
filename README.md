# WSMF
## tl;dr
This package contains implementations of two novel approaches to warm-starting encoder-based warm-start of the Bayesian Hyperparameter Optimization. It allows both training and using meta-models which can help in this meta-task.


## Contents
**Meta-models** - As for now it contains two approaches to encoder-based warm-start:

* Metric learning (`Dataset2VecMetricLearning`) - As an encoder it uses Dataset2Vec which is trained in a way that it produces representations whose distances to each other correspond to distances of the landmarkers (vectors of performances of a predefined set of hyperparameter configuration)
* Landmarker reconstruction (`LandmarkerReconstructionTrainingInterface`) - As an encoder it uses Dataset2Vec which produces a latent representation of the entire dataset (of any size) and passes it to MLP which outputs predictions of the landmarker vector

**Selectors** - for usage for this intended meta-task `wsmf` provides API to use encoder for proposing hyperparameter configuration. It contains the following samplers:

* Selector which is choosing based on the learned representation that is applicable in the metric learning approach (`RepresentationBasedHpSelector`)
* Selector which is based on the reconstructed landmarkers (`ReconstructionBasedHpSelector`)
* Random selector from the predefined portfolio (`RandomHpSelector`)
* Selector which chooses the best configuration on average (`RankBasedHpSelector`)
* Selector that chooses configurations based on the vector of landmarkers itself (`LandmarkerHpSelector`)

## Examples of usage
**Training metric learning based meta-model**
```Python
# tensors X, y are torch.Tensor objects which correspond to feature and target matrices
train_datasets = { # training meta-dataset
    "dataset_train_1": (tensor_X1, tensor_y1),
    "dataset_train_2": (tensor_X2, tensor_y2),
    ...
}
val_datasets = { # validation meta-dataset
    "dataset_val_1": (tensor_X1, tensor_y1),
    "dataset_val_2": (tensor_X2, tensor_y2),
    ...
}

# tensors l1, l2, .. corresponds to vector of landmarkers in torch.Tensor format
train_landmarkers = { # training meta-dataset
    "dataset_train_1": l1,
    "dataset_train_2": l2,
    ...
}
val_landmarkers = { # validation meta-dataset
    "dataset_val_1": l1,
    "dataset_val_2": l2,
    ...
}

train_dataset = EncoderHpoDataset(train_datasets, train_landmarkers)
train_dataloader = EncoderMetricLearningLoader(train_dataset, train_num_batches, train_batch_size)
val_dataset = EncoderHpoDataset(val_datasets, val_landmarkers)
val_dataloader = EncoderMetricLearningLoader(val_dataset, val_num_batches, val_batch_size)
val_dataloader = GenericRepeatableDataLoader(val_dataloader) # Loader which produces repeatable batches

model = Dataset2VecMetricLearning()
trainer = pl.Trainer()
trainer.fit(model, train_loader, val_loader)
```

**Using selector based on reconstruction**
```Python
datasets = { # datasets to search from (in this case used for closest dataset search)
    "dataset_1": (tensor_X1, tensor_y1),
    "dataset_2": (tensor_X2, tensor_y2),
    ...
}
landmarkers = { # landmarkers to search from (is this case used for proposing best configurations)
    "dataset_val_1": l1,
    "dataset_val_2": l2,
    ...
}
configurations = [
    {"hp1": val1, "hp2": val2},
    {"hp1": val3, "hp2": val4},
    ...
]

meta_model = Dataset2VecForLandmarkerReconstruction.load_from_checkpoint("path_to_meta_model.ckpt")
selector = ReconstructionBasedHpSelector(
    meta_model,
    datasets,
    landmarkers,
    configurations
)
# Usage
new_dataset = (X, y) # torch.Tensor
n_configurations = 10
configurations = selector.propose_configurations(new_dataset, n_configurations)
```


## Development
Commands useful during development:
* Seting env variables - `export PYTHONPATH=(backtick)pwd(backtick)`
* Install dependencies - `pip install -r requirements_dev.txt`
* To run unit tests - `pytest`
* Check code quality - `./scripts/check_code.sh`
* Relase - `python -m build && twine upload dist/*`
