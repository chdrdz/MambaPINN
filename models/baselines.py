from __future__ import annotations

import torch
from torch import nn

from .mamba_pinn import MambaPINN


class BaselineHead(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([h, t], dim=-1)))


class MLPBaseline(nn.Module):
    def __init__(self, window_size: int, d_model: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(window_size, d_model), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(max(n_layers - 1, 0)):
            layers.extend([nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*layers)
        self.head = BaselineHead(d_model, dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is None:
            t = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        h = self.encoder(x)
        return self.head(h, t)


class RecurrentBaseline(nn.Module):
    def __init__(
            self,
            cell_type: str,
            d_model: int = 64,
            n_layers: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        rnn_dropout = dropout if n_layers > 1 else 0.0
        if cell_type == "rnn":
            self.encoder = nn.RNN(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=rnn_dropout)
        elif cell_type == "lstm":
            self.encoder = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=rnn_dropout)
        elif cell_type == "gru":
            self.encoder = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError(f"Unsupported recurrent baseline: {cell_type}")
        self.norm = nn.LayerNorm(d_model)
        self.head = BaselineHead(d_model, dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is None:
            t = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        seq = self.input_proj(x.unsqueeze(-1))
        out, _ = self.encoder(seq)
        h = self.norm(out[:, -1, :])
        return self.head(h, t)


class TransformerBaseline(nn.Module):
    def __init__(
            self,
            window_size: int,
            d_model: int = 64,
            n_layers: int = 2,
            dropout: float = 0.1,
            n_heads: int = 4,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = BaselineHead(d_model, dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is None:
            t = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        seq = self.input_proj(x.unsqueeze(-1)) + self.pos_embedding[:, : x.size(1), :]
        h = self.norm(self.encoder(seq)[:, -1, :])
        return self.head(h, t)


def build_baseline_model(
        model_name: str,
        window_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        backend: str = "simple",
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLPBaseline(window_size=window_size, d_model=d_model, n_layers=n_layers, dropout=dropout)
    if model_name in {"rnn", "lstm", "gru"}:
        return RecurrentBaseline(cell_type=model_name, d_model=d_model, n_layers=n_layers, dropout=dropout)
    if model_name == "transformer":
        return TransformerBaseline(window_size=window_size, d_model=d_model, n_layers=n_layers, dropout=dropout)
    if model_name == "mamba_only":
        return MambaPINN(
            window_size=window_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            backend=backend,
        )
    raise ValueError(f"Unsupported baseline model: {model_name}")


BASELINE_MODEL_NAMES = ("mlp", "rnn", "lstm", "gru", "transformer", "mamba_only")
