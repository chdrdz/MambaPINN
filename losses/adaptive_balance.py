from __future__ import annotations

import torch
from torch import nn


class AdaptiveBalanceLoss(nn.Module):
    def __init__(self, use_boundary: bool = True):
        super().__init__()
        self.log_sigma_pred = nn.Parameter(torch.zeros(()))
        self.log_sigma_physics = nn.Parameter(torch.zeros(()))
        self.log_sigma_boundary = nn.Parameter(torch.zeros(()))
        self.use_boundary = use_boundary

    def forward(self, pred_loss: torch.Tensor, physics_loss: torch.Tensor,
                boundary_loss: torch.Tensor | None = None) -> torch.Tensor:
        loss = torch.exp(-self.log_sigma_pred) * pred_loss + self.log_sigma_pred
        loss = loss + torch.exp(-self.log_sigma_physics) * physics_loss + self.log_sigma_physics
        if self.use_boundary and boundary_loss is not None:
            loss = loss + torch.exp(-self.log_sigma_boundary) * boundary_loss + self.log_sigma_boundary
        return loss

    def weights_dict(self) -> dict[str, float]:
        return {
            "w_pred": float(torch.exp(-self.log_sigma_pred).detach().cpu()),
            "w_physics": float(torch.exp(-self.log_sigma_physics).detach().cpu()),
            "w_boundary": float(torch.exp(-self.log_sigma_boundary).detach().cpu()),
        }
