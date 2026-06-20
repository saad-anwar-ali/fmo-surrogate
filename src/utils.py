"""
utils.py — Unit conversions, physics helpers, plotting utilities, and
density-matrix operations for the 7-site FMO surrogate project.

Unit conventions throughout this codebase
------------------------------------------
  _cm   → wavenumbers (cm^-1)
  _fs   → femtoseconds
  _rf   → rad/fs  (angular frequency, the natural QuTiP unit for time in fs)
  _K    → Kelvin

Conversion:  omega_rf = E_cm * 2*pi*c   where c = 2.998e-5 cm/fs
"""

import os, random, yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import torch

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_CM_PER_FS: float = 2.99792458e-5          # speed of light, cm fs^-1
HBAR_CM_FS:  float = 1.0 / (2.0*np.pi*C_CM_PER_FS)  # ≈ 5308.8 cm^-1 fs
K_B_CM_PER_K: float = 0.6950356             # Boltzmann constant, cm^-1 K^-1


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def cm_to_rad_per_fs(energy_cm: float) -> float:
    """Convert wavenumbers (cm^-1) to angular frequency (rad/fs)."""
    return float(energy_cm) * 2.0 * np.pi * C_CM_PER_FS

def rad_per_fs_to_cm(omega_rf: float) -> float:
    """Convert angular frequency (rad/fs) to wavenumbers (cm^-1)."""
    return float(omega_rf) / (2.0 * np.pi * C_CM_PER_FS)

def dephasing_rate_cm(T_K: float, lambda_cm: float,
                      omega_c_cm: float, alpha_scale: float) -> float:
    """
    Pure dephasing rate from the Ohmic Drude-Lorentz bath
    (high-T Markovian limit, Ishizaki & Fleming 2009):

        gamma_phi = 2 * alpha * lambda * k_B * T / (hbar * omega_c)

    All in cm^-1 units.
    """
    return alpha_scale * (2.0 * lambda_cm * K_B_CM_PER_K * T_K) / (HBAR_CM_FS * omega_c_cm)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 7-site density matrix helpers
# The 7x7 complex density matrix is flattened to 98 real numbers:
#   rho_flat[0:49]  = Re(rho)  row-major
#   rho_flat[49:98] = Im(rho)  row-major
# ---------------------------------------------------------------------------

N_SITES = 7
N_RHO   = N_SITES * N_SITES * 2  # 98

def rho_matrix_to_flat(rho_matrix: np.ndarray) -> np.ndarray:
    """
    Flatten a complex NxN density matrix to 2*N^2 real numbers.
    Convention: [Re(rho).ravel(), Im(rho).ravel()]

    Works for any square matrix; the 7-site project uses N=7 → shape (98,).
    """
    n2 = rho_matrix.shape[-2] * rho_matrix.shape[-1]
    shape = rho_matrix.shape[:-2]
    re = rho_matrix.real.reshape(*shape, n2)
    im = rho_matrix.imag.reshape(*shape, n2)
    return np.concatenate([re, im], axis=-1)

def rho_flat_to_matrix(rho_flat: np.ndarray, n_sites: int = N_SITES) -> np.ndarray:
    """Reconstruct complex density matrix from flattened real representation."""
    n2 = n_sites * n_sites
    re = rho_flat[..., :n2].reshape(*rho_flat.shape[:-1], n_sites, n_sites)
    im = rho_flat[..., n2:].reshape(*rho_flat.shape[:-1], n_sites, n_sites)
    return re + 1j * im

def extract_populations(rho_flat: np.ndarray, n_sites: int = N_SITES) -> np.ndarray:
    """
    Extract site populations (diagonal elements of Re(rho)).
    Returns shape (..., n_sites).
    Diagonal indices of NxN row-major matrix: 0, N+1, 2N+2, ...
    """
    diag_idx = np.arange(n_sites) * (n_sites + 1)   # 0, 8, 16, 24, 32, 40, 48
    return rho_flat[..., diag_idx]

def extract_coherence_12(rho_flat: np.ndarray, n_sites: int = N_SITES) -> np.ndarray:
    """
    Magnitude of off-diagonal coherence |rho_{01}| (between sites 1 and 2).
    Re(rho_01) is at flat index 1; Im(rho_01) at n_sites^2 + 1.
    """
    n2 = n_sites * n_sites
    re = rho_flat[..., 1]
    im = rho_flat[..., n2 + 1]
    return np.sqrt(re**2 + im**2)

def compute_trace(rho_flat: np.ndarray, n_sites: int = N_SITES) -> np.ndarray:
    """Tr(rho) = sum of Re(rho_jj)."""
    return extract_populations(rho_flat, n_sites).sum(axis=-1)


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Lightweight CSV training logger (no external deps)."""

    def __init__(self, log_path: str, fieldnames: List[str]) -> None:
        self.log_path   = log_path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(",".join(fieldnames) + "\n")

    def log(self, metrics: dict) -> None:
        row = [str(metrics.get(k, "")) for k in self.fieldnames]
        with open(self.log_path, "a") as f:
            f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "lines.linewidth": 1.8,
        "axes.grid": True, "grid.alpha": 0.3,
    })

def save_figure(fig, name: str, figure_dir: str, dpi: int = 150) -> None:
    os.makedirs(figure_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figure_dir, f"{name}.{ext}"),
                    dpi=dpi if ext == "png" else None, bbox_inches="tight")

def param_label(T_K, lambda_cm, omega_c_cm, alpha, init_site=None) -> str:
    s = f"T={T_K:.0f}K λ={lambda_cm:.0f} ωc={omega_c_cm:.0f} α={alpha:.1f}"
    if init_site is not None:
        s += f" site{int(init_site)+1}"
    return s
