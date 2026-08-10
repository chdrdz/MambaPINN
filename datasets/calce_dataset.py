from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BatterySeries, normalize_soh


CALCE_CELL_MAP: dict[str, tuple[str, ...]] = {
    "CS2_35": ("CS2_35",),
    "CS2_36": ("CS2_36",),
    "CS2_37": ("CS2_37",),
    "CS2_38": ("CS2_38",),
}


def _extract_channel_sheet(xlsx: pd.ExcelFile) -> str:
    for sheet in xlsx.sheet_names:
        if sheet.startswith("Channel_"):
            return sheet
    raise ValueError(f"未在 xlsx 中找到 Channel_* 工作表：{xlsx.io}")


def _read_capacity_series(path: Path) -> np.ndarray:
    xls = pd.ExcelFile(path)
    sheet = _extract_channel_sheet(xls)
    df = pd.read_excel(path, sheet_name=sheet, usecols=["Cycle_Index", "Discharge_Capacity(Ah)"])
    if df.empty:
        return np.asarray([], dtype=np.float32)
    df = df.dropna(subset=["Cycle_Index", "Discharge_Capacity(Ah)"])
    if df.empty:
        return np.asarray([], dtype=np.float32)

    cycle_max = (
        df.groupby("Cycle_Index", sort=True)["Discharge_Capacity(Ah)"]
        .max()
        .sort_index()
        .to_numpy(dtype=np.float32)
    )
    if cycle_max.size == 0:
        return np.asarray([], dtype=np.float32)

    # CALCE 里该列是随循环累积的容量记录，因此使用相邻循环差分得到单次放电容量。
    capacity = np.diff(np.concatenate(([0.0], cycle_max.astype(np.float32))))
    capacity = capacity[np.isfinite(capacity)]
    capacity = capacity[(capacity > 0.2) & (capacity < 2.5)]
    return capacity.astype(np.float32)


def _cache_path(raw_dir: Path) -> Path:
    return raw_dir / ".calce_capacity_cache.npz"


def _load_capacity_cache(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {key: data[key].astype(np.float32) for key in data.files}


def _save_capacity_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **values)


def _extract_all_capacities(raw_dir: Path) -> dict[str, np.ndarray]:
    cached = _load_capacity_cache(_cache_path(raw_dir))
    if cached:
        print(f"[calce] loaded capacity cache: {_cache_path(raw_dir)}")
        return cached

    values: dict[str, np.ndarray] = {}
    cell_dirs = [p for p in sorted(raw_dir.iterdir()) if p.is_dir()]
    if not cell_dirs:
        raise ValueError(f"CALCE 数据目录下没有电池子目录：{raw_dir}")

    for cell_dir in cell_dirs:
        xlsx_files = sorted(cell_dir.glob("*.xlsx"))
        print(f"[calce] extracting {cell_dir.name}: {len(xlsx_files)} xlsx files")
        caps: list[float] = []
        for xlsx in xlsx_files:
            try:
                cap = _read_capacity_series(xlsx)
            except Exception:
                continue
            if cap.size == 0:
                continue
            caps.extend(cap.tolist())

        if caps:
            values[cell_dir.name] = np.asarray(caps, dtype=np.float32)

    if not values:
        raise ValueError(f"CALCE 数据目录中没有提取出可用容量序列：{raw_dir}")

    _save_capacity_cache(_cache_path(raw_dir), values)
    print(f"[calce] saved capacity cache: {_cache_path(raw_dir)}")
    return values


def load_calce_series(raw_dir: str | Path, rated_capacity: float | None = 2.0) -> list[BatterySeries]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"CALCE 数据目录不存在：{raw_dir}")

    capacities = _extract_all_capacities(raw_dir)
    series = [
        BatterySeries(cell_id=cell_id, soh=normalize_soh(capacity, rated_capacity))
        for cell_id, capacity in sorted(capacities.items())
    ]
    if not series:
        raise ValueError(f"CALCE 数据目录中没有提取出可用容量序列：{raw_dir}")
    return series