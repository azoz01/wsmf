import os
from pathlib import Path


class PathsProvider:

    def __init__(self, root_data_path: Path, root_results_path: Path):
        self.root_data_path = root_data_path
        self.hp_selectors_path = root_data_path / "selectors_config.yaml"
        self.tasks_ids_path = root_data_path / "tasks.json"
        self.prohibited_datasets_path = (
            root_data_path / "prohibited_datasets.json"
        )
        self.raw_datasets_path = root_data_path / "datasets_raw"
        self.datasets_binarized_path = root_data_path / "datasets_binarized"
        self.datasets_splitted_path = root_data_path / "datasets_splitted"
        self.train_meta_dataset_path = (
            root_data_path / "datasets_meta" / "train"
        )
        self.val_meta_dataset_path = root_data_path / "datasets_meta" / "val"
        self.train_meta_dataset_path_for_plain_d2v = (
            root_data_path / "datasets_meta_plain_d2v" / "train"
        )
        self.val_meta_dataset_path_for_plain_d2v = (
            root_data_path / "datasets_meta_plain_d2v" / "val"
        )

        self.root_results_path = root_results_path
        self.hp_portfolio_configurations_path = (
            root_results_path / "hp_portfolio_configurations"
        )
        self.metafeatures_path = root_results_path / "metafeatures.parquet"
        self.best_hpo_path = root_results_path / "best_hpo"
        self.landmarkers_path = root_results_path / "landmarkers"
        self.warmstart_results_path = root_results_path / "warmstart_results"
        self.encoders_results_path = root_results_path / "encoder"
        self.meta_dataset_analysis_path = (
            root_results_path / "metadataset_analysis"
        )
        self.results_analysis_path = root_results_path / "results_analysis"

        for field_name in dir(self):
            attr_val = getattr(self, field_name)
            if isinstance(attr_val, Path):
                if attr_val.name == attr_val.stem:
                    attr_val.mkdir(exist_ok=True, parents=True)


subpath = os.environ.get("EXPERIMENTS_SUBPATH", "openml")
paths_provider = PathsProvider(
    Path(f"data/{subpath}"), Path(f"results/{subpath}")
)
