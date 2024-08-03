#!/bin/bash
set -e

export PATHS_TO_CHECK="wsmf experiments_engine test"

echo "Running isort"
isort --profile=black --line-length=79 $PATHS_TO_CHECK

echo "Running black"
black --line-length=79 $PATHS_TO_CHECK

echo "Running flake8"
flake8 --ignore=W605,W503 --exclude experiments_engine/cd_plot.py $PATHS_TO_CHECK

echo "Running mypy"
mypy \
    --install-types \
    --non-interactive \
    --ignore-missing-imports \
    --strict \
    --namespace-packages \
    --exclude experiments_engine/cd_plot.py \
    $PATHS_TO_CHECK
