from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    text = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    print(f"{prefix}: {text}")