from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .base import BatterySeries, list_data_files, normalize_soh


CAPACITY_KEYS = ("capacity", "Capacity", "Qd", "qd", "QDischarge", "discharge_capacity", "cycle_life")
MIT_STANDARD_BATCH_FILES = (
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
)


def _find_numeric_sequence(obj: Any, keys: tuple[str, ...] = CAPACITY_KEYS) -> np.ndarray | None:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                seq = _find_numeric_sequence(obj[key], keys)
                if seq is not None:
                    return seq
        for value in obj.values():
            seq = _find_numeric_sequence(value, keys)
            if seq is not None and len(seq) > 10:
                return seq
    if isinstance(obj, pd.DataFrame):
        for key in keys:
            if key in obj.columns:
                return obj[key].to_numpy(dtype=np.float32)
        numeric_cols = obj.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            return obj[numeric_cols[0]].to_numpy(dtype=np.float32)
    if isinstance(obj, (list, tuple)):
        for value in obj:
            seq = _find_numeric_sequence(value, keys)
            if seq is not None and len(seq) > 10:
                return seq
    arr = np.asarray(obj) if not isinstance(obj, (str, bytes, Path)) else None
    if arr is not None and np.issubdtype(arr.dtype, np.number) and arr.size > 10:
        return arr.astype(np.float32).reshape(-1)
    return None


def _read_hdf5_summary_capacity(h5_file: h5py.File, summary_ref: Any) -> np.ndarray | None:
    summary = h5_file[summary_ref]
    if "QDischarge" not in summary:
        return None
    capacity = np.asarray(summary["QDischarge"][()]).astype(np.float32).reshape(-1)
    capacity = capacity[np.isfinite(capacity)]
    return capacity if capacity.size > 10 else None


def _load_hdf5_batch_capacities(path: Path) -> list[tuple[str, np.ndarray]]:
    cells: list[tuple[str, np.ndarray]] = []
    with h5py.File(path, "r") as f:
        if "batch" not in f or "summary" not in f["batch"]:
            return cells
        summary_refs = f["batch"]["summary"]
        for idx in range(summary_refs.shape[0]):
            capacity = _read_hdf5_summary_capacity(f, summary_refs[idx, 0])
            if capacity is None:
                continue
            cells.append((f"{path.stem}_cell{idx:03d}", capacity))
    return cells


def _to_series(cells: list[tuple[str, np.ndarray]], rated_capacity: float | None) -> list[BatterySeries]:
    series: list[BatterySeries] = []
    for cell_id, capacity in cells:
        soh = normalize_soh(capacity, rated_capacity)
        series.append(BatterySeries(cell_id=cell_id, soh=soh))
    return series


def _load_hdf5_batch_series(path: Path, rated_capacity: float | None) -> list[BatterySeries]:
    return _to_series(_load_hdf5_batch_capacities(path), rated_capacity)


def _drop_one_based(items: list[tuple[str, np.ndarray]], indices: list[int]) -> list[tuple[str, np.ndarray]]:
    drop = {idx - 1 for idx in indices}
    return [item for idx, item in enumerate(items) if idx not in drop]


def _load_official_mit_series(
    raw_dir: Path,
    rated_capacity: float | None,
    remove_unfinished_batch1: bool = True,
) -> list[BatterySeries] | None:
    paths = [raw_dir / name for name in MIT_STANDARD_BATCH_FILES]
    # 只有三个 MIT 官方 batch 文件齐全时才进入标准过滤分支；否则退回通用文件读取逻辑。
    if not all(path.exists() for path in paths):
        return None

    batch1 = _load_hdf5_batch_capacities(paths[0])
    batch2 = _load_hdf5_batch_capacities(paths[1])
    batch3 = _load_hdf5_batch_capacities(paths[2])
    if not batch1 or not batch2 or not batch3:
        return None

    # 官方 LoadData.m：Batch 2 的 5 块续测电池追加到 Batch 1 前 5 块，而不是作为新电池计数。
    batch2_continuation_indices = [8, 9, 10, 16, 17]
    for batch1_idx, batch2_one_based_idx in enumerate(batch2_continuation_indices):
        batch2_idx = batch2_one_based_idx - 1
        if batch1_idx >= len(batch1) or batch2_idx >= len(batch2):
            continue
        cell_id, capacity = batch1[batch1_idx]
        _, continuation_capacity = batch2[batch2_idx]
        batch1[batch1_idx] = (cell_id, np.concatenate([capacity, continuation_capacity]))
    batch2 = _drop_one_based(batch2, batch2_continuation_indices)

    # 官方 LoadData.m：先删除 Batch 3 第 38 块异常通道，再删除未达到容量终止阈值的电池。
    batch3 = _drop_one_based(batch3, [38])
    batch3 = [cell for cell in batch3 if float(cell[1][-1]) <= 0.885]

    # 官方 LoadData.m：删除 noisy Batch 8 batteries，索引基于前面过滤后的 Batch 3。
    batch3 = _drop_one_based(batch3, [3, 40, 41])

    combined = batch1 + batch2 + batch3
    if remove_unfinished_batch1:
        # 这里得到当前项目采用的 MIT 标准 124 块集合；不删除原始文件，只在加载阶段过滤。
        combined = _drop_one_based(combined, [9, 11, 13, 14, 23])

    if not combined:
        raise ValueError(f"MIT 标准三批次数据过滤后为空：{raw_dir}")
    return _to_series(combined, rated_capacity)


def _load_one_file(path: Path) -> np.ndarray | None:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        seq = _find_numeric_sequence(obj)
    elif suffix == ".mat":
        try:
            obj = loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            return None
        seq = _find_numeric_sequence({k: v for k, v in obj.items() if not k.startswith("__")})
    elif suffix == ".csv":
        seq = _find_numeric_sequence(pd.read_csv(path))
    elif suffix == ".npy":
        seq = _find_numeric_sequence(np.load(path, allow_pickle=True))
    elif suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        seq = _find_numeric_sequence({k: npz[k] for k in npz.files})
    else:
        raise ValueError(f"不支持的 MIT 数据格式：{path}")
    return seq


def load_mit_series(raw_dir: str | Path, rated_capacity: float | None = 1.1) -> list[BatterySeries]:
    raw_path = Path(raw_dir)
    # 优先按 MIT 官方三批次规则加载，保证训练、验证、测试都基于同一套标准电池集合。
    official_series = _load_official_mit_series(raw_path, rated_capacity)
    if official_series is not None:
        return official_series

    # 兼容非官方整理数据：逐文件寻找容量序列，便于后续扩展或临时实验。
    files = list_data_files(raw_path, (".pkl", ".pickle", ".mat", ".csv", ".npy", ".npz"))
    series: list[BatterySeries] = []
    skipped: list[str] = []
    for path in files:
        loaded_hdf5 = False
        if path.suffix.lower() == ".mat":
            try:
                hdf5_series = _load_hdf5_batch_series(path, rated_capacity)
            except OSError:
                hdf5_series = []
            if hdf5_series:
                series.extend(hdf5_series)
                loaded_hdf5 = True
        if loaded_hdf5:
            continue

        capacity = _load_one_file(path)
        if capacity is None:
            skipped.append(path.name)
            continue
        soh = normalize_soh(capacity, rated_capacity)
        series.append(BatterySeries(cell_id=path.stem, soh=soh))
    if skipped:
        warnings.warn(f"MIT 数据中有 {len(skipped)} 个文件未能识别容量序列，已跳过：{skipped}")
    if not series:
        raise ValueError(f"MIT 数据目录中没有可用电池序列：{raw_dir}")
    return sorted(series, key=lambda item: item.cell_id)