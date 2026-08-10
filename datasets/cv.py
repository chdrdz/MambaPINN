from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader, random_split

from .base import BatterySeries, SOHWindowDataset


@dataclass(frozen=True)
class CellFold:
    fold_id: int
    train_series: list[BatterySeries]
    test_series: list[BatterySeries]
    train_cell_ids: list[str]
    test_cell_ids: list[str]
    valid_series: list[BatterySeries] | None = None
    valid_cell_ids: list[str] | None = None


def build_leave_one_out_folds(series: Sequence[BatterySeries]) -> list[CellFold]:
    ordered = list(series)
    if len(ordered) != 4:
        raise ValueError(f"4 折电池留一验证预期 4 块电池，当前为 {len(ordered)} 块：{[item.cell_id for item in ordered]}")

    folds: list[CellFold] = []
    for fold_id, test_item in enumerate(ordered, start=1):
        train_series = [item for item in ordered if item.cell_id != test_item.cell_id]
        folds.append(
            CellFold(
                fold_id=fold_id,
                train_series=train_series,
                test_series=[test_item],
                train_cell_ids=[item.cell_id for item in train_series],
                test_cell_ids=[test_item.cell_id],
            )
        )
    return folds


def build_cell_validation_folds(series: Sequence[BatterySeries]) -> list[CellFold]:
    ordered = list(series)
    if len(ordered) != 4:
        raise ValueError(f"12 组电池级验证预期 4 块电池，当前为 {len(ordered)} 块：{[item.cell_id for item in ordered]}")

    folds: list[CellFold] = []
    fold_id = 1
    for test_item in ordered:
        candidates = [item for item in ordered if item.cell_id != test_item.cell_id]
        for valid_item in candidates:
            train_series = [item for item in candidates if item.cell_id != valid_item.cell_id]
            folds.append(
                CellFold(
                    fold_id=fold_id,
                    train_series=train_series,
                    valid_series=[valid_item],
                    test_series=[test_item],
                    train_cell_ids=[item.cell_id for item in train_series],
                    valid_cell_ids=[valid_item.cell_id],
                    test_cell_ids=[test_item.cell_id],
                )
            )
            fold_id += 1
    return folds


def split_window_dataset(
    series: Sequence[BatterySeries],
    window_size: int,
    stride: int,
    valid_ratio: float,
    seed: int,
):
    dataset = SOHWindowDataset(series, window_size=window_size, stride=stride)
    total = len(dataset)
    if total < 2:
        raise ValueError(
            f"训练数据窗口过少，无法继续划分验证集：total={total}, window_size={window_size}, stride={stride}"
        )

    valid_size = int(round(total * valid_ratio))
    valid_size = max(1, min(valid_size, total - 1))
    train_size = total - valid_size
    generator = torch.Generator().manual_seed(int(seed))
    train_subset, valid_subset = random_split(dataset, [train_size, valid_size], generator=generator)
    return train_subset, valid_subset


def build_fold_loaders(
    train_series: Sequence[BatterySeries],
    test_series: Sequence[BatterySeries],
    window_size: int,
    stride: int,
    batch_size: int,
    valid_ratio: float,
    seed: int,
    shuffle_train: bool = True,
):
    train_ds, valid_ds = split_window_dataset(
        train_series,
        window_size=window_size,
        stride=stride,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    test_ds = SOHWindowDataset(test_series, window_size=window_size, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    stats = {
        "train_windows": len(train_ds),
        "valid_windows": len(valid_ds),
        "test_windows": len(test_ds),
    }
    return train_loader, valid_loader, test_loader, stats


def build_cell_validation_loaders(
    train_series: Sequence[BatterySeries],
    valid_series: Sequence[BatterySeries],
    test_series: Sequence[BatterySeries],
    window_size: int,
    stride: int,
    batch_size: int,
    shuffle_train: bool = True,
):
    train_ds = SOHWindowDataset(train_series, window_size=window_size, stride=stride)
    valid_ds = SOHWindowDataset(valid_series, window_size=window_size, stride=stride)
    test_ds = SOHWindowDataset(test_series, window_size=window_size, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    stats = {
        "train_windows": len(train_ds),
        "valid_windows": len(valid_ds),
        "test_windows": len(test_ds),
    }
    return train_loader, valid_loader, test_loader, stats