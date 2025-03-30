#!/bin/bash
export PYTHONPATH=`pwd`
export EXPERIMENTS_SUBPATH=openml

python bin/load_data/download_data_from_openml.py
python bin/load_data/preprocess_data.py
python bin/metatrain_preparation/perform_hpo_on_all_datasets.py
python bin/metatrain_preparation/calculate_metafeatures.py
python bin/load_data/meta_split.py
python bin/metatrain_preparation/select_hp_for_portfolio.py --objective XGBoostObjective --model-name xgboost
python bin/metatrain_preparation/calculate_landmarkers.py --objective XGBoostObjective --model-name xgboost
python bin/train_meta_models/train_dataset2vec_base.py
python bin/train_meta_models/train_dataset2vec_metric.py
python bin/train_meta_models/train_dataset2vec_reconstruction.py
python bin/analysis/analyze_distance_correlation.py
python bin/evaluate/perform_warmstart_experiment.py --objective XGBoostObjective --model-name xgboost --sampler-name TPESampler
python bin/analysis/analyze_warmstart.py
python bin/analysis/analyze_configurations_distances.py