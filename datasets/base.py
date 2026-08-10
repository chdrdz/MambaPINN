from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BatterySeries:
    cell_id: str
    soh: np.ndarray


def build_windows(series: np.ndarray, window_size: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    if len(series) <= window_size:
        return (
            np.empty((0, window_size), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )
    xs, ys, ts = [], [], []
    # 用前 window_size 个 SOH 点预测下一个 SOH 点；t 归一化到 [0, 1] 供 PINN 物理项求导。
    denom = max(len(series) - 1, 1)
    for start in range(0, len(series) - window_size, stride):
        target_idx = start + window_size
        xs.append(series[start:target_idx])
        ys.append(series[target_idx])
        ts.append(target_idx / denom)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32).reshape(-1, 1),
        np.asarray(ts, dtype=np.float32).reshape(-1, 1),
    )


class SOHWindowDataset(Dataset):
    def __init__(self, series_list: Iterable[BatterySeries], window_size: int, stride: int = 1):
        xs, ys, ts, cell_ids = [], [], [], []
        for item in series_list:
            # 每块电池独立切窗口，避免跨电池拼接造成不真实的时序样本。
            x, y, t = build_windows(item.soh, window_size, stride)
            if len(x) == 0:
                continue
            xs.append(x)
            ys.append(y)
            ts.append(t)
            cell_ids.extend([item.cell_id] * len(x))
        if not xs:
            raise ValueError("没有构造出任何窗口样本，请检查数据长度、window_size 和数据路径。")
        self.x = torch.from_numpy(np.concatenate(xs, axis=0))
        self.y = torch.from_numpy(np.concatenate(ys, axis=0))
        self.t = torch.from_numpy(np.concatenate(ts, axis=0))
        self.cell_ids = cell_ids

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {"x": self.x[idx], "y": self.y[idx], "t": self.t[idx], "cell_id": self.cell_ids[idx]}


def split_series(series: list[BatterySeries], train_count: int, valid_count: int) -> tuple[list[BatterySeries], list[BatterySeries], list[BatterySeries]]:
    if len(series) < train_count + valid_count + 1:
        raise ValueError(f"电池数量不足：当前 {len(series)}，无法按 train={train_count}, valid={valid_count} 划分。")
    # 按电池整体切分，而不是按窗口随机切分；测试电池不会出现在训练窗口里。
    train = series[:train_count]
    valid = series[train_count : train_count + valid_count]
    test = series[train_count + valid_count :]
    return train, valid, test


def normalize_soh(capacity: np.ndarray, rated_capacity: float | None = None) -> np.ndarray:
    capacity = np.asarray(capacity, dtype=np.float32).reshape(-1)
    capacity = capacity[np.isfinite(capacity)]
    if len(capacity) == 0:
        raise ValueError("容量序列为空或全为非有限值。")
    # 优先使用数据集配置的额定容量；未配置时退回首圈容量，便于临时数据也能跑通。
    denom = float(rated_capacity) if rated_capacity else float(capacity[0])
    if denom <= 0:
        raise ValueError(f"额定容量必须为正数，当前为 {denom}。")
    soh = capacity / denom
    # 上限保留到 1.2，兼容少量实测容量高于标称容量的早期循环。
    return np.clip(soh, 0.0, 1.2).astype(np.float32)


def list_data_files(raw_dir: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{raw_dir}")
    files: list[Path] = []
    for suffix in suffixes:
        files.extend(raw_dir.rglob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"未在 {raw_dir} 下找到数据文件，支持后缀：{suffixes}")
    return sorted(files)