import json
from pathlib import Path

import torch

device = "cuda" if torch.cuda.is_available else "cpu"


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_dataset_name_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.stem
