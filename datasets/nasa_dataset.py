from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .base import BatterySeries, list_data_files, normalize_soh


NASA_CELLS = ("B0005", "B0006", "B0007", "B0018")


def _mat_struct_get(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _extract_capacity_from_cycle(cycle_obj: Any) -> np.ndarray:
    capacities: list[float] = []
    cycles = np.ravel(cycle_obj)
    for cyc in cycles:
        cycle_type = _mat_struct_get(cyc, "type")
        if isinstance(cycle_type, np.ndarray):
            cycle_type = str(cycle_type.item()) if cycle_type.size == 1 else str(cycle_type)
        if cycle_type != "discharge":
            continue
        data = _mat_struct_get(cyc, "data")
        cap = _mat_struct_get(data, "Capacity")
        if cap is None:
            continue
        arr = np.asarray(cap, dtype=np.float32).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            capacities.append(float(arr[0]))
    return np.asarray(capacities, dtype=np.float32)


def _load_nasa_file(path: Path) -> np.ndarray:
    obj = loadmat(path, squeeze_me=True, struct_as_record=False)
    root_key = path.stem
    root = obj.get(root_key)
    if root is None:
        candidates = [v for k, v in obj.items() if not k.startswith("__")]
        if not candidates:
            raise ValueError(f"NASA mat 文件无有效内容：{path}")
        root = candidates[0]
    cycle = _mat_struct_get(root, "cycle")
    if cycle is None:
        raise ValueError(f"无法在 {path} 中找到 cycle 字段。")
    capacity = _extract_capacity_from_cycle(cycle)
    if len(capacity) == 0:
        raise ValueError(f"未能从 {path} 提取 discharge Capacity。")
    return capacity


def load_nasa_series(raw_dir: str | Path, rated_capacity: float | None = 2.0) -> list[BatterySeries]:
    files = list_data_files(raw_dir, (".mat",))
    series: list[BatterySeries] = []
    for path in files:
        if path.stem not in NASA_CELLS:
            continue
        capacity = _load_nasa_file(path)
        soh = normalize_soh(capacity, rated_capacity)
        series.append(BatterySeries(cell_id=path.stem, soh=soh))
    if not series:
        raise ValueError(f"NASA 数据目录中没有可用的 B0005/B0006/B0007/B0018 .mat 文件：{raw_dir}")
    return sorted(series, key=lambda item: item.cell_id)