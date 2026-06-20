"""
pinn_model.py  —  Physics-informed feedforward network for 7-site FMO.

Input:  [params_norm (5), t_norm (1)]  →  6-dim
Output: rho_flat (98)  —  flattened 7×7 density matrix

Physics penalties
-----------------
  L_trace:    (Tr(rho) - 1)^2               →  trace conservation
  L_pos:      Σ_i ReLU(-rho_ii)^2           →  population positivity
  L_herm_off: Σ_{i≠j} (rho_ij - rho_ji*)^2  →  Hermiticity of off-diagonal
              Note: in the real representation Re(rho_ij)=Re(rho_ji) and
              Im(rho_ij)=-Im(rho_ji), so imaginary diagonals should be zero.
"""

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import N_RHO, N_SITES


class PINNSurrogate(nn.Module):
    """Physics-informed feedforward surrogate for 7-site FMO."""

    _ACTS = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}

    def __init__(self, n_input=6, n_rho=N_RHO,
                 hidden_sizes: List[int] = None,
                 activation: str = "tanh", dropout: float = 0.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 512, 512]
        act_cls = self._ACTS[activation]
        layers, in_d = [], n_input
        for h in hidden_sizes:
            layers += [nn.Linear(in_d, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, n_rho))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_physics_loss(rho_pred: Tensor,
                         lambda_trace: float = 10.0,
                         lambda_pos: float = 5.0,
                         lambda_herm: float = 2.0) -> Tensor:
    """
    Soft physics penalties encoding quantum mechanical constraints on ρ.

    For a valid density matrix ρ must satisfy:
      (a) Tr(ρ) = 1                          — normalisation
      (b) ρᵢᵢ ≥ 0  ∀ i                       — populations non-negative
      (c) ρ = ρ†  (Hermiticity), which means:
            Re(ρᵢⱼ) =  Re(ρⱼᵢ)   ∀ i,j      — real part symmetric
            Im(ρᵢⱼ) = -Im(ρⱼᵢ)   ∀ i,j      — imaginary part antisymmetric
            Im(ρᵢᵢ) = 0           ∀ i        — imaginary diagonal vanishes

    The previous implementation only checked Im(ρᵢᵢ) = 0 and missed the
    off-diagonal symmetry conditions Re(ρᵢⱼ) = Re(ρⱼᵢ) and Im(ρᵢⱼ) = -Im(ρⱼᵢ).
    This is now fully enforced.

    Parameters
    ----------
    rho_pred : Tensor, shape (B, 98)
        Predicted flattened density matrix.
        Convention: [Re(ρ) row-major (49 elems) | Im(ρ) row-major (49 elems)]
    lambda_trace : float   Weight for trace penalty.
    lambda_pos   : float   Weight for positivity penalty.
    lambda_herm  : float   Weight for Hermiticity penalty.

    Returns
    -------
    Tensor, scalar — total physics penalty.
    """
    n2 = N_SITES * N_SITES   # 49
    re = rho_pred[:, :n2].reshape(-1, N_SITES, N_SITES)   # (B, 7, 7)
    im = rho_pred[:, n2:].reshape(-1, N_SITES, N_SITES)   # (B, 7, 7)

    # --- 1. Trace conservation: Tr(Re(ρ)) = 1 ---
    diag_re    = re.diagonal(dim1=1, dim2=2)                    # (B, 7)
    trace_loss = lambda_trace * ((diag_re.sum(dim=-1) - 1.0)**2).mean()

    # --- 2. Population positivity: Re(ρᵢᵢ) ≥ 0 ---
    pos_loss   = lambda_pos * (F.relu(-diag_re)**2).sum(dim=-1).mean()

    # --- 3. Full Hermiticity: ρ = ρ† ---
    #
    # (a) Imaginary diagonal must vanish: Im(ρᵢᵢ) = 0
    diag_im       = im.diagonal(dim1=1, dim2=2)               # (B, 7)
    herm_diag     = (diag_im**2).sum(dim=-1).mean()
    #
    # (b) Real part must be symmetric: Re(ρᵢⱼ) - Re(ρⱼᵢ) = 0
    #     re - re.transpose(1,2) should be zero for all i,j
    re_antisym    = re - re.transpose(1, 2)                    # (B, 7, 7)
    herm_re       = (re_antisym**2).sum(dim=(-1, -2)).mean()
    #
    # (c) Imaginary part must be antisymmetric: Im(ρᵢⱼ) + Im(ρⱼᵢ) = 0
    #     im + im.transpose(1,2) should be zero for all i,j
    im_sym        = im + im.transpose(1, 2)                    # (B, 7, 7)
    herm_im       = (im_sym**2).sum(dim=(-1, -2)).mean()

    herm_loss     = lambda_herm * (herm_diag + herm_re + herm_im)

    return trace_loss + pos_loss + herm_loss


def compute_total_loss(rho_pred, rho_true,
                       lambda_trace=10.0, lambda_pos=5.0, lambda_herm=2.0):
    mse  = F.mse_loss(rho_pred, rho_true)
    phys = compute_physics_loss(rho_pred, lambda_trace, lambda_pos, lambda_herm)
    return mse + phys, mse, phys


def build_pinn_from_config(cfg):
    mc = cfg["model"]
    n_input = mc["n_params"] + 1   # 5 params + 1 time = 6
    m = PINNSurrogate(n_input=n_input, n_rho=mc["n_rho_elements"], **mc["pinn"])
    print(f"PINN: {m.count_parameters():,} parameters")
    return m
