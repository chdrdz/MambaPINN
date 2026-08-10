from __future__ import annotations

import torch
from torch import nn

from .mamba import MambaBackbone


class MambaPINN(nn.Module):
    def __init__(
            self,
            window_size: int,
            d_model: int = 64,
            n_layers: int = 2,
            dropout: float = 0.1,
            backend: str = "simple",
            d_state: int = 16,
            expand_factor: int = 4,
            d_conv: int = 4,
            dt_rank: int | str = "auto",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
    ):
        super().__init__()
        # 单变量 SOH 窗口先映射到 d_model 维，后续 Mamba 才能在通道空间建模退化模式。
        self.input_proj = nn.Linear(1, d_model)
        self.backbone = MambaBackbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            backend=backend,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
        )
        self.norm = nn.LayerNorm(d_model)
        # 预测头额外接收归一化时间 t，让数据分支和物理约束使用同一个时间坐标。
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"输入 x 期望形状为 (batch, window_size)，当前为 {tuple(x.shape)}")
        seq = x.unsqueeze(-1)
        h = self.input_proj(seq)
        h = self.backbone(h)
        # 只取窗口末端隐藏状态，语义上等价于“基于历史窗口预测下一圈 SOH”。
        h = self.norm(h[:, -1, :])
        if t is None:
            # 推理接口允许不传 t；此时只使用数据分支，物理项训练时仍会显式传入 t。
            t = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        out = self.head(torch.cat([h, t], dim=-1))
        # 输出限制在 [0, 1]，与 SOH 归一化目标保持一致，边界 loss 额外兜底异常值。
        return torch.sigmoid(out)
