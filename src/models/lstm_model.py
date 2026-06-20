"""
lstm_model.py  —  LSTM autoregressive surrogate for 7-site FMO dynamics.

Input at each step:  [params_norm (5), rho_flat_t (98)]  → 103-dim
Output at each step: rho_flat_{t+1}  →  98-dim

Rollout: autoregressively from rho(0) given exact initial condition.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import N_RHO


class LSTMSurrogate(nn.Module):
    """
    LSTM autoregressive surrogate for 7-site Lindblad dynamics.

    Architecture
    ------------
    Input projection  →  LSTM (num_layers, hidden_size)  →  Output projection

    Teacher forcing at train time; full autoregressive rollout at inference.
    """

    def __init__(self, n_params=5, n_rho=N_RHO,
                 hidden_size=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.n_params    = n_params
        self.n_rho       = n_rho
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        in_dim = n_params + n_rho   # 5 + 98 = 103
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.Tanh())
        self.lstm       = nn.LSTM(hidden_size, hidden_size, num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.out_drop   = nn.Dropout(dropout)
        self.out_proj   = nn.Linear(hidden_size, n_rho)

    def forward(self, params: Tensor, rho_sequence: Tensor) -> Tensor:
        """
        Teacher-forcing forward pass.
        params:       (B, n_params)
        rho_sequence: (B, T, n_rho)
        Returns:      (B, T-1, n_rho)  — predictions for steps 1..T-1
        """
        B, T, _ = rho_sequence.shape
        params_exp = params.unsqueeze(1).expand(-1, T-1, -1)
        x = torch.cat([params_exp, rho_sequence[:, :-1, :]], dim=-1)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.out_proj(self.out_drop(out))

    @torch.no_grad()
    def rollout(self, params: Tensor, rho0: Tensor, n_steps: int) -> Tensor:
        """
        Autoregressive rollout from exact initial condition.
        params: (B, n_params)   rho0: (B, n_rho)
        Returns (B, n_steps, n_rho) including rho0 at index 0.
        """
        self.eval()
        B = params.shape[0]
        h = torch.zeros(self.num_layers, B, self.hidden_size, device=params.device)
        c = torch.zeros(self.num_layers, B, self.hidden_size, device=params.device)
        traj  = [rho0.unsqueeze(1)]
        rho_t = rho0
        for _ in range(n_steps - 1):
            x = torch.cat([params, rho_t], dim=-1).unsqueeze(1)
            x = self.input_proj(x)
            out, (h, c) = self.lstm(x, (h, c))
            rho_t = self.out_proj(self.out_drop(out.squeeze(1)))
            traj.append(rho_t.unsqueeze(1))
        return torch.cat(traj, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm_from_config(cfg):
    mc = cfg["model"]
    m  = LSTMSurrogate(n_params=mc["n_params"], n_rho=mc["n_rho_elements"],
                       **mc["lstm"])
    print(f"LSTM: {m.count_parameters():,} parameters")
    return m
