from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from .base import BatterySeries, SOHWindowDataset, split_series
from .calce_dataset import load_calce_series
from .mit_dataset import load_mit_series
from .nasa_dataset import load_nasa_series
from .sohbenchmark_mit_dataset import load_sohbenchmark_mit_series


def load_series_by_name(dataset: str, raw_dir: str | Path, rated_capacity: float | None):
    dataset = dataset.lower()
    if dataset == "mit":
        return load_mit_series(raw_dir, rated_capacity=rated_capacity)
    if dataset == "sohbenchmark_mit":
        return load_sohbenchmark_mit_series(raw_dir, rated_capacity=rated_capacity)
    if dataset == "nasa":
        return load_nasa_series(raw_dir, rated_capacity=rated_capacity)
    if dataset == "calce":
        return load_calce_series(raw_dir, rated_capacity=rated_capacity)
    raise ValueError(f"暂不支持的数据集：{dataset}")


def get_rated_capacity(config: dict, dataset: str) -> float | None:
    dataset = dataset.lower()
    capacities = config.get("rated_capacity", {})
    if isinstance(capacities, dict):
        value = capacities.get(dataset)
        return float(value) if value is not None else None
    legacy_value = config.get("preprocess", {}).get("rated_capacity")
    return float(legacy_value) if legacy_value is not None else None


def build_dataloaders(config: dict):
    dataset_name = config["experiment"]["dataset"]
    paths = config["paths"]
    preprocess = config["preprocess"]
    train_cfg = config["train"]

    raw_dir = paths[f"{dataset_name}_raw_dir"]
    rated_capacity = get_rated_capacity(config, dataset_name)
    series = load_series_by_name(dataset_name, raw_dir, rated_capacity)
    train_series, valid_series, test_series = split_series(
        series,
        train_count=int(config["experiment"]["train_cells"]),
        valid_count=int(config["experiment"]["valid_cells"]),
    )

    common = {
        "window_size": int(preprocess["window_size"]),
        "stride": int(preprocess.get("stride", 1)),
    }
    train_ds = SOHWindowDataset(train_series, **common)
    valid_ds = SOHWindowDataset(valid_series, **common)
    test_ds = SOHWindowDataset(test_series, **common)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    valid_loader = DataLoader(valid_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)
    return train_loader, valid_loader, test_loader, {"train": train_series, "valid": valid_series, "test": test_series}


def build_target_loader(config: dict, target: str):
    paths = config["paths"]
    preprocess = config["preprocess"]
    raw_dir = paths[f"{target}_raw_dir"]
    rated_capacity = get_rated_capacity(config, target)
    series = load_series_by_name(target, raw_dir, rated_capacity)
    dataset = SOHWindowDataset(
        series,
        window_size=int(preprocess["window_size"]),
        stride=int(preprocess.get("stride", 1)),
    )
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]), shuffle=False)
    return loader, series