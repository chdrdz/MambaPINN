from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> torch.Tensor:
    x = torch.tensor(float(value)).clamp_min(1e-8)
    return torch.log(torch.expm1(x))


class VerhulstPhysicsLoss(nn.Module):
    def __init__(
            self,
            r_init: float = 0.01,
            c_max_init: float = 1.0,
            c_min_init: float = 0.7,
            eps: float = 1e-6,
            monotonic_weight: float = 0.1,
            tail_steps: int = 0,
    ):
        super().__init__()
        # 用可学习 raw 参数再映射到合法区间，训练时不会出现负退化率或非法容量边界。
        self.raw_r = nn.Parameter(_inverse_softplus(float(r_init)))
        self.raw_c_max = nn.Parameter(_inverse_softplus(float(c_max_init)))
        self.raw_c_min = nn.Parameter(torch.logit(torch.tensor(float(c_min_init)).clamp(0.01, 0.99)))
        self.eps = eps
        self.monotonic_weight = float(monotonic_weight)
        self.tail_steps = int(tail_steps)

    @property
    def r(self) -> torch.Tensor:
        return F.softplus(self.raw_r)

    @property
    def c_max(self) -> torch.Tensor:
        return F.softplus(self.raw_c_max)

    @property
    def c_min(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_c_min)

    def forward(self, pred: torch.Tensor, t: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        capacity_range = torch.clamp(self.c_max - self.c_min, min=self.eps)
        if history is not None:
            if history.dim() != 2:
                raise ValueError(f"history 期望形状为 (batch, window_size)，当前为 {tuple(history.shape)}")
            history = history.to(device=pred.device, dtype=pred.dtype)
            tail = history[:, -max(self.tail_steps, 1):] if self.tail_steps > 0 else history
            trajectory = torch.cat([tail, pred], dim=1)
            delta_soh = trajectory[:, 1:] - trajectory[:, :-1]
            soh_mid = trajectory[:, :-1]
            verhulst_rhs = -self.r * (soh_mid - self.c_min) * (1.0 - (soh_mid - self.c_min) / capacity_range)
            residual = delta_soh - verhulst_rhs
            physics = torch.mean(residual ** 2)
            monotonic = torch.mean(torch.relu(delta_soh) ** 2)
            return physics + self.monotonic_weight * monotonic

        if not t.requires_grad:
            return pred.new_tensor(0.0)
        grad = torch.autograd.grad(
            pred,
            t,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        if grad is None:
            # 如果某次模型输出没有依赖 t，物理项退化为 0，避免训练直接中断。
            return pred.new_tensor(0.0)
        # SOH 随 cycle 增加应整体下降，因此 Verhulst 项使用负号表示退化方向。
        verhulst_rhs = -self.r * (pred - self.c_min) * (1.0 - (pred - self.c_min) / capacity_range)
        residual = grad - verhulst_rhs
        return torch.mean(residual ** 2)

    def parameters_dict(self) -> dict[str, float]:
        return {
            "r": float(self.r.detach().cpu()),
            "c_max": float(self.c_max.detach().cpu()),
            "c_min": float(self.c_min.detach().cpu()),
        }


def boundary_loss(pred: torch.Tensor) -> torch.Tensor:
    lower = torch.relu(-pred)
    upper = torch.relu(pred - 1.2)
    return torch.mean(lower ** 2 + upper ** 2)
