from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from losses import CompositePINNLoss
from utils import ensure_dir, regression_metrics, save_json

@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_loss(
        model: nn.Module,
        batch: dict[str, Any],
        loss_module: CompositePINNLoss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = batch["x"].float()
    y = batch["y"].float()
    t = batch["t"].float()
    # 有限差分物理约束直接使用历史窗口和预测点，不再依赖模型必须显式使用 t。
    pred = model(x, t)
    return loss_module(pred, y, t, history=x)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    sample_ids: list[Any] = []
    cell_ids: list[Any] = []

    for batch in loader:
        batch = move_batch(batch, device)
        x = batch["x"].float()
        y = batch["y"].float()
        t = batch["t"].float()
        pred = model(x, t)
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        ts.append(t.detach().cpu().numpy())
        batch_sample_ids = batch.get("sample_id")
        if batch_sample_ids is not None:
            if torch.is_tensor(batch_sample_ids):
                sample_ids.extend(batch_sample_ids.detach().cpu().numpy().reshape(-1).tolist())
            else:
                sample_ids.extend(list(batch_sample_ids))
        batch_cell_ids = batch.get("cell_id")
        if batch_cell_ids is not None:
            if isinstance(batch_cell_ids, str):
                cell_ids.extend([batch_cell_ids])
            else:
                cell_ids.extend(list(batch_cell_ids))

    y_true = np.concatenate(ys).reshape(-1)
    y_pred = np.concatenate(preds).reshape(-1)
    t_all = np.concatenate(ts).reshape(-1)
    metrics = regression_metrics(y_true, y_pred)
    outputs: dict[str, Any] = {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "t": t_all.tolist(),
    }
    if sample_ids:
        outputs["sample_id"] = sample_ids
    if cell_ids:
        outputs["cell_id"] = cell_ids
    return metrics, outputs


def save_prediction_csv(outputs: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    t_values = outputs.get("t", [None] * len(y_true))
    sample_ids = outputs.get("sample_id", list(range(len(y_true))))
    cell_ids = outputs.get("cell_id", [""] * len(y_true))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "cell_id", "t", "y_true", "y_pred", "error", "abs_error"],
        )
        writer.writeheader()
        for idx, (actual, predicted) in enumerate(zip(y_true, y_pred)):
            error = float(predicted) - float(actual)
            writer.writerow(
                {
                    "sample_id": sample_ids[idx] if idx < len(sample_ids) else idx,
                    "cell_id": cell_ids[idx] if idx < len(cell_ids) else "",
                    "t": t_values[idx] if idx < len(t_values) else "",
                    "y_true": actual,
                    "y_pred": predicted,
                    "error": error,
                    "abs_error": abs(error),
                }
            )
    return path


@torch.no_grad()
def evaluate_autoregressive(
        model: nn.Module,
        series_list,
        window_size: int,
        device: torch.device,
        horizon: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    all_true: list[float] = []
    all_pred: list[float] = []
    per_cell: list[dict[str, Any]] = []

    for cell_idx, item in enumerate(series_list, start=1):
        soh = np.asarray(item.soh, dtype=np.float32).reshape(-1)
        if len(soh) <= window_size:
            print(f"[autoregressive] skip cell={item.cell_id} len={len(soh)} <= window_size={window_size}")
            continue
        steps = len(soh) - window_size
        if horizon is not None:
            steps = min(steps, int(horizon))
        if steps <= 0:
            print(f"[autoregressive] skip cell={item.cell_id} steps={steps}")
            continue

        print(
            f"[autoregressive] cell {cell_idx}/{len(series_list)} "
            f"id={item.cell_id} steps={steps} window_size={window_size}"
        )
        window = soh[:window_size].copy()
        preds: list[float] = []
        denom = max(len(soh) - 1, 1)
        progress_interval = max(1, steps // 10)
        for step in range(steps):
            target_idx = window_size + step
            x = torch.from_numpy(window.reshape(1, -1)).to(device=device, dtype=torch.float32)
            t = torch.tensor([[target_idx / denom]], device=device, dtype=torch.float32)
            pred = float(model(x, t).detach().cpu().reshape(-1)[0])
            preds.append(pred)
            window = np.concatenate([window[1:], np.asarray([pred], dtype=np.float32)]).astype(np.float32)
            if (step + 1) % progress_interval == 0 or step + 1 == steps:
                print(f"[autoregressive] cell={item.cell_id} step={step + 1}/{steps}")

        true = soh[window_size: window_size + steps].astype(np.float32)
        cell_metrics = regression_metrics(true, np.asarray(preds, dtype=np.float32))
        per_cell.append(
            {
                "cell_id": item.cell_id,
                "steps": int(steps),
                "metrics": cell_metrics,
                "y_true": true.tolist(),
                "y_pred": preds,
            }
        )
        all_true.extend(true.tolist())
        all_pred.extend(preds)

    if not all_true:
        raise ValueError("没有可用于自回归评估的序列，请检查 window_size、horizon 和数据长度。")
    metrics = regression_metrics(np.asarray(all_true, dtype=np.float32), np.asarray(all_pred, dtype=np.float32))
    return metrics, {"y_true": all_true, "y_pred": all_pred, "per_cell": per_cell}


def train_one_epoch(
        model: nn.Module,
        loader,
        optimizer: torch.optim.Optimizer,
        loss_module: CompositePINNLoss,
        device: torch.device,
        grad_clip: float = 1.0,
) -> EpochResult:
    model.train()
    loss_module.train()
    total_objective = 0.0
    total_pred = 0.0
    total_physics = 0.0
    total_boundary = 0.0
    total_n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        # 显式调用组合损失，保证预测误差、物理残差和边界约束在同一处反传。
        loss, logs = forward_loss(model, batch, loss_module)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_module.parameters()), grad_clip)
        optimizer.step()
        batch_n = len(batch["x"])
        total_objective += float(logs["total"]) * batch_n
        total_pred += float(logs["pred"]) * batch_n
        total_physics += float(logs["physics"]) * batch_n
        total_boundary += float(logs["boundary"]) * batch_n
        total_n += batch_n
    denom = max(total_n, 1)
    metrics = {
        "objective": total_objective / denom,
        "pred_loss": total_pred / denom,
        "physics_loss": total_physics / denom,
        "boundary_loss": total_boundary / denom,
    }
    return EpochResult(loss=metrics["objective"], metrics=metrics)


def fit(
        model: nn.Module,
        train_loader,
        valid_loader,
        device: torch.device,
        epochs: int,
        lr: float,
        weight_decay: float,
        checkpoint_dir: str | Path,
        grad_clip: float = 1.0,
        patience: int = 50,
        use_physics: bool = True,
        use_boundary: bool = True,
        use_adaptive_balance: bool = True,
        lambda_physics: float = 1.0,
        lambda_boundary: float = 0.1,
        physics_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    # loss_module 含有可学习的物理参数/自适应权重，因此必须和模型参数一起优化和保存。
    loss_module = CompositePINNLoss(
        physics_config=physics_config,
        use_physics=use_physics,
        use_boundary=use_boundary,
        use_adaptive_balance=use_adaptive_balance,
        lambda_physics=lambda_physics,
        lambda_boundary=lambda_boundary,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    best_metric = float("inf")
    best_path = checkpoint_dir / "best_model.pt"
    best_state = None
    history: list[dict[str, float]] = []
    wait = 0

    print(
        f"[train] start | epochs={epochs} lr={lr} weight_decay={weight_decay} "
        f"patience={patience} grad_clip={grad_clip} device={device}"
    )
    print(
        f"[train] loss | physics={use_physics} boundary={use_boundary} adaptive={use_adaptive_balance} "
        f"lambda_physics={lambda_physics} lambda_boundary={lambda_boundary}"
    )

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_module,
            device,
            grad_clip=grad_clip,
        )
        valid_metrics, _ = evaluate(model, valid_loader, device)
        record = {
            "epoch": float(epoch),
            "train_objective": float(train_result.metrics["objective"]),
            "train_pred_loss": float(train_result.metrics["pred_loss"]),
            "train_physics_loss": float(train_result.metrics["physics_loss"]),
            "train_boundary_loss": float(train_result.metrics["boundary_loss"]),
            "valid_rmse": float(valid_metrics["rmse"]),
            "valid_mae": float(valid_metrics["mae"]),
            "valid_mape": float(valid_metrics["mape"]),
            "valid_r2": float(valid_metrics["r2"]),
            **loss_module.diagnostics(),
        }
        history.append(record)
        improved = valid_metrics["rmse"] < best_metric
        if improved:
            # 以验证集 RMSE 选择 checkpoint，测试集只在 evaluate.py 中使用，避免训练阶段碰测试集。
            best_metric = valid_metrics["rmse"]
            best_state = {
                "model": model.state_dict(),
                "loss": loss_module.state_dict(),
                "loss_config": loss_module.extra_state_dict(),
                "epoch": epoch,
                "valid_metrics": valid_metrics,
            }
            torch.save(best_state, best_path)
            wait = 0
        else:
            wait += 1

        print(
            f"[train] epoch {epoch:03d}/{epochs:03d} | "
            f"objective={train_result.metrics['objective']:.6f} "
            f"pred_loss={train_result.metrics['pred_loss']:.6e} "
            f"physics_loss={train_result.metrics['physics_loss']:.6e} "
            f"boundary_loss={train_result.metrics['boundary_loss']:.6e} "
            f"valid_rmse={valid_metrics['rmse']:.6f} "
            f"valid_mae={valid_metrics['mae']:.6f} "
            f"valid_r2={valid_metrics['r2']:.6f} "
            f"best_rmse={best_metric:.6f} "
            f"wait={wait}"
        )
        if improved:
            print(f"[train] new best checkpoint saved: {best_path} | valid_rmse={best_metric:.6f}")
        if wait >= patience:
            # 连续 patience 轮验证集没有改善就停止，减少过拟合和无意义长跑。
            print(f"[train] early stopping triggered at epoch {epoch}, patience={patience}")
            break

    if best_state is None:
        # 极端情况下没有任何一轮被标记为 improved，也要保存最后状态保证后续 evaluate 有入口。
        best_state = {
            "model": model.state_dict(),
            "loss": loss_module.state_dict(),
            "loss_config": loss_module.extra_state_dict(),
            "epoch": epochs,
            "valid_metrics": valid_metrics,
        }
        torch.save(best_state, best_path)

    save_json({"history": history, "best": best_state["valid_metrics"]}, checkpoint_dir / "history.json")
    print(f"[train] finished | best_checkpoint={best_path} best_valid_metrics={best_state['valid_metrics']}")
    return {"best_path": str(best_path), "history": history, "best_valid_metrics": best_state["valid_metrics"]}
