from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    eps = 1e-8
    err = y_pred - y_true
    mse = np.mean(err ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum(err ** 2) / max(denom, eps))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}