from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .pscan import pscan


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: int | str = "auto"
    d_state: int = 16
    expand_factor: int = 4
    d_conv: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True

    def __post_init__(self) -> None:
        # d_inner 是 Mamba block 内部展开后的宽度，决定 SSM 和卷积分支的通道数。
        self.d_inner = self.expand_factor * self.d_model
        # dt_rank 控制时间步长 delta 的低秩表示维度，auto 时按 d_model 自适应取值。
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 逐层执行 residual Mamba block，再做最终归一化，输出形状保持为 (B, L, D)。
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual 结构：先归一化，再让 mixer 建模，最后加回原始输入。
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        dt_rank = int(config.dt_rank)

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.x_proj = nn.Linear(config.d_inner, dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, config.d_inner, bias=True)

        # delta 初始化在较小正区间内，避免训练初期状态转移过快导致序列记忆不稳定。
        dt_init_std = dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {config.dt_init}")

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        # A_log/D 是 SSM 的长期记忆核心参数：A 管状态衰减，D 保留输入直达路径。
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        # 输入投影一分为二：x 进入卷积+SSM 主分支，z 作为门控分支调制输出。
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # depthwise causal conv 只在每个通道内部做局部时序混合，并裁掉右侧 padding。
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)
        z = F.silu(z)
        return self.out_proj(y * z)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        # A 用负指数参数化，保证离散状态转移整体稳定；D 是输入到输出的直连项。
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        dt_rank = int(self.config.dt_rank)

        # 每个时间步动态生成 delta/B/C，这是 selective SSM 区别于固定 RNN 的核心。
        delta_b_c = self.x_proj(x)
        delta, B, C = torch.split(delta_b_c, [dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        if self.config.pscan:
            return self.selective_scan(x, delta, A, B, C, D)
        return self.selective_scan_seq(x, delta, A, B, C, D)

    @staticmethod
    def selective_scan(
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        delta_a = torch.exp(delta.unsqueeze(-1) * A)
        delta_b = delta.unsqueeze(-1) * B.unsqueeze(2)
        bx = delta_b * x.unsqueeze(-1)
        hs = pscan(delta_a, bx)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def selective_scan_seq(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        delta_a = torch.exp(delta.unsqueeze(-1) * A)
        delta_b = delta.unsqueeze(-1) * B.unsqueeze(2)
        bx = delta_b * x.unsqueeze(-1)

        h = torch.zeros(
            x.size(0),
            self.config.d_inner,
            self.config.d_state,
            device=x.device,
            dtype=x.dtype,
        )
        states = []
        for t in range(x.size(1)):
            h = delta_a[:, t] * h + bx[:, t]
            states.append(h)
        hs = torch.stack(states, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x


class MambaBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float = 0.0,
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
        if backend not in {"auto", "simple", "mamba"}:
            raise ValueError(f"当前仅支持内置简化 Mamba backend，收到 backend={backend}")
        self.backend_name = "simple_mamba"
        self.dropout = nn.Dropout(dropout)
        self.block = Mamba(
            MambaConfig(
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                expand_factor=expand_factor,
                d_conv=d_conv,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                pscan=True,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))