from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .base import BatterySeries, list_data_files, normalize_soh


def _get_mat_field(obj: Any, field: str) -> Any:
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, np.ndarray) and obj.dtype.names and field in obj.dtype.names:
        return obj[field]
    if isinstance(obj, dict):
        return obj.get(field)
    return None


def _extract_capacity_from_cycles(cycles: Any) -> np.ndarray:
    capacities: list[float] = []
    flattened = np.ravel(cycles)
    for cycle in flattened:
        capacity = _get_mat_field(cycle, "capacity")
        if capacity is None:
            continue
        arr = np.asarray(capacity, dtype=np.float32).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            capacities.append(float(arr[0]))
    return np.asarray(capacities, dtype=np.float32)


def _load_group_file(path: Path, rated_capacity: float | None) -> list[BatterySeries]:
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    battery = mat.get("battery")
    if battery is None:
        raise ValueError(f"SOHbenchmark MIT 文件缺少 battery 字段：{path}")

    series: list[BatterySeries] = []
    cells = np.ravel(battery)
    for idx, cell in enumerate(cells, start=1):
        cycles = _get_mat_field(cell, "cycles")
        if cycles is None:
            continue
        capacity = _extract_capacity_from_cycles(cycles)
        if capacity.size == 0:
            continue
        soh = normalize_soh(capacity, rated_capacity)
        series.append(BatterySeries(cell_id=f"{path.stem}_cell{idx}", soh=soh))
    return series


def load_sohbenchmark_mit_series(raw_dir: str | Path, rated_capacity: float | None = 1.1) -> list[BatterySeries]:
    """Load SOHbenchmark's preprocessed Toyota-MIT-Stanford subset.

    The SOHbenchmark MIT data stores each group as one .mat file under
    data/MIT/charge. Each file contains multiple batteries, and each battery
    contains cycle-level capacity labels. This loader extracts only the capacity
    degradation sequence so it can share the same SOH pipeline as other datasets.
    """
    raw_dir = Path(raw_dir)
    charge_dir = raw_dir / "charge" if (raw_dir / "charge").exists() else raw_dir
    files = list_data_files(charge_dir, (".mat",))

    series: list[BatterySeries] = []
    for path in files:
        series.extend(_load_group_file(path, rated_capacity))

    if not series:
        raise ValueError(f"SOHbenchmark MIT 数据目录中没有可用容量序列：{raw_dir}")
    return sorted(series, key=lambda item: item.cell_id)