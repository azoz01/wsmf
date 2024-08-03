#!/bin/bash
export PYTHONPATH=`pwd`
export EXPERIMENTS_SUBPATH=openml

python bin/load_data/download_data_from_openml.py
python bin/load_data/preprocess_data.py
python bin/metatrain_preparation/perform_hpo_on_all_datasets.py
python bin/metatrain_preparation/calculate_metafeatures.py
python bin/load_data/meta_split.py
python bin/metatrain_preparation/select_hp_for_portfolio.py --objective XGBoostObjective --model-name xgboost