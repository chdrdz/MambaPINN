from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .adaptive_balance import AdaptiveBalanceLoss
from .pinn_loss import VerhulstPhysicsLoss, boundary_loss


class CompositePINNLoss(nn.Module):
    def __init__(
            self,
            physics_config: dict[str, float] | None = None,
            use_physics: bool = True,
            use_boundary: bool = True,
            use_adaptive_balance: bool = True,
            lambda_physics: float = 1.0,
            lambda_boundary: float = 0.1,
    ):
        super().__init__()
        physics_config = physics_config or {}
        self.use_physics = use_physics
        self.use_boundary = use_boundary
        self.use_adaptive_balance = use_adaptive_balance
        self.lambda_physics = float(lambda_physics)
        self.lambda_boundary = float(lambda_boundary)

        self.physics = VerhulstPhysicsLoss(
            r_init=float(physics_config.get("r_init", 0.01)),
            c_max_init=float(physics_config.get("c_max_init", 1.0)),
            c_min_init=float(physics_config.get("c_min_init", 0.7)),
            eps=float(physics_config.get("eps", 1e-6)),
            monotonic_weight=float(physics_config.get("monotonic_weight", 0.1)),
            tail_steps=int(physics_config.get("tail_steps", 0)),
        )
        self.balance = AdaptiveBalanceLoss(use_boundary=use_boundary)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor, history: torch.Tensor | None = None) -> tuple[
        torch.Tensor, dict[str, torch.Tensor]]:
        # pred_loss 负责贴近实测 SOH；physics/boundary 负责把曲线限制在合理退化形态内。
        pred_loss = torch.mean((pred - target) ** 2)
        physics_loss = self.physics(pred, t, history=history) if self.use_physics else pred_loss.new_tensor(0.0)
        bnd_loss = boundary_loss(pred) if self.use_boundary else pred_loss.new_tensor(0.0)

        # adaptive 模式下权重由可学习不确定性参数给出；关闭时使用配置里的固定 lambda。
        if self.use_adaptive_balance:
            total = self.balance(pred_loss, physics_loss, bnd_loss if self.use_boundary else None)
        else:
            total = pred_loss + self.lambda_physics * physics_loss + self.lambda_boundary * bnd_loss

        logs = {
            "total": total.detach(),
            "pred": pred_loss.detach(),
            "physics": physics_loss.detach(),
            "boundary": bnd_loss.detach(),
        }
        return total, logs

    def diagnostics(self) -> dict[str, float]:
        values: dict[str, float] = {
            **{f"physics_{k}": v for k, v in self.physics.parameters_dict().items()},
            "lambda_physics": self.lambda_physics,
            "lambda_boundary": self.lambda_boundary,
        }
        if self.use_adaptive_balance:
            values.update({f"weight_{k}": v for k, v in self.balance.weights_dict().items()})
        return values

    def extra_state_dict(self) -> dict[str, Any]:
        return {
            "use_physics": self.use_physics,
            "use_boundary": self.use_boundary,
            "use_adaptive_balance": self.use_adaptive_balance,
            "lambda_physics": self.lambda_physics,
            "lambda_boundary": self.lambda_boundary,
        }
