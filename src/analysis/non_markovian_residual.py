"""
non_markovian_residual.py  —  Non-Markovian / vibronic fingerprint analysis.

Core idea
---------
The LSTM surrogate is trained exclusively on Lindblad (Markovian) data.
When evaluated on HEOM (non-Markovian) trajectories, the prediction residual

    R(t; params) = rho_HEOM(t; params) - rho_LSTM(t; params)

is a direct, data-driven fingerprint of non-Markovian and vibronic effects
that the Lindblad approximation cannot capture. This is publishable because:

  (a) It provides a model-free characterisation of memory effects —
      no assumption about the form of non-Markovianity is required.

  (b) The residual has a clear physical interpretation:
        - Large |R_coherence(t)| at early t → bath memory extends coherence
        - Large |R_pop7(t)|               → non-Markovian correction to RC yield
        - Temperature scaling of ||R||     → identifies the quantum-classical crossover

  (c) It demonstrates the limits of the Markovian surrogate quantitatively,
      which is directly relevant to the vibronic coherence debate
      (Duan 2017, Cao 2020, Scholes 2017) discussed in the companion paper.

Figures produced
----------------
  1. residual_landscape.png
       Heatmap of ||R||_F (Frobenius norm of residual matrix) vs (T, λ)
       at t=200 fs and t_end.  Shows where non-Markovian effects are largest.

  2. residual_trajectories.png
       Time traces of R_{coherence}(t) = |ρ12_HEOM| - |ρ12_LSTM| and
       R_{RC}(t) = P7_HEOM - P7_LSTM for three (T, λ) regimes.
       The sign and timescale of R_coherence reveals memory-enhanced coherence.

  3. non_markovian_signature.png
       ||R||_F vs T at fixed λ (shows quantum→classical crossover).
       At low T memory effects are largest; at high T Markov becomes accurate.
"""

import os, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (load_config, set_plot_style, save_figure,
                   extract_populations, extract_coherence_12,
                   rho_flat_to_matrix, N_RHO, N_SITES)
from dataset import ParamNormaliser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lstm(cfg, device):
    from models.lstm_model import build_lstm_from_config
    path = os.path.join(cfg["training"]["checkpoint_dir"], "lstm_best.pt")
    ckpt = torch.load(path, map_location=device)
    m = build_lstm_from_config(cfg).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, ParamNormaliser.from_dict(ckpt["normaliser"])


@torch.no_grad()
def _predict_lstm(model, norm, params_raw, rho0, n_steps, device):
    """Single trajectory LSTM prediction."""
    pn = norm.transform(params_raw.astype(np.float32))
    p  = torch.tensor(pn,   dtype=torch.float32, device=device).unsqueeze(0)
    r  = torch.tensor(rho0, dtype=torch.float32, device=device).unsqueeze(0)
    traj = model.rollout(p, r, n_steps).squeeze(0).cpu().numpy()
    return traj   # (n_steps, 98)


def _frobenius_residual(rho_heom, rho_lstm):
    """
    Per-timestep Frobenius norm of the residual density matrix.

    rho_heom, rho_lstm : (n_steps, 98)
    Returns : (n_steps,)
    """
    n_steps = rho_heom.shape[0]
    norms = np.zeros(n_steps)
    for t in range(n_steps):
        R = rho_flat_to_matrix(rho_heom[t] - rho_lstm[t])
        norms[t] = float(np.linalg.norm(R, "fro"))
    return norms


def _coherence_residual(rho_heom, rho_lstm):
    """Signed coherence residual: |ρ12_HEOM| - |ρ12_LSTM|."""
    return extract_coherence_12(rho_heom) - extract_coherence_12(rho_lstm)


def _rc_residual(rho_heom, rho_lstm):
    """RC-proximal (site 7) population residual: P7_HEOM - P7_LSTM."""
    return (extract_populations(rho_heom)[:, 6] -
            extract_populations(rho_lstm)[:, 6])


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_non_markovian_analysis(cfg, device=None, save_dir=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_dir is None:
        save_dir = cfg["evaluation"]["figure_dir"]

    heom_path = cfg["data"]["heom_hdf5"]
    if not os.path.exists(heom_path):
        print("HEOM dataset not found — skipping non-Markovian residual analysis.")
        return {}

    set_plot_style()
    print("Loading LSTM surrogate...")
    model, norm = _load_lstm(cfg, device)

    # --- Load HEOM data ---
    with h5py.File(heom_path, "r") as f:
        h_params = f["trajectories/params"][:]
        h_rho    = f["trajectories/rho_flat"][:]
        h_t_fs   = f["trajectories/t_fs"][:]

    n_heom  = h_params.shape[0]
    n_steps = len(h_t_fs)

    # Lindblad time grid (training window) — normalised to match HEOM length
    l_t_fs = np.linspace(0.0, h_t_fs[-1], n_steps)

    print(f"  Computing LSTM predictions for {n_heom} HEOM trajectories...")
    lstm_preds = np.zeros_like(h_rho)
    for i in range(n_heom):
        lstm_preds[i] = _predict_lstm(
            model, norm, h_params[i], h_rho[i, 0], n_steps, device)

    # --- Residual stats per trajectory ---
    frob_norms = np.zeros((n_heom, n_steps))
    coh_resids = np.zeros((n_heom, n_steps))
    rc_resids  = np.zeros((n_heom, n_steps))

    for i in range(n_heom):
        frob_norms[i] = _frobenius_residual(h_rho[i], lstm_preds[i])
        coh_resids[i] = _coherence_residual(h_rho[i], lstm_preds[i])
        rc_resids[i]  = _rc_residual(h_rho[i], lstm_preds[i])

    # Mean residual at t_end per (T, λ) combination
    T_vals   = np.unique(h_params[:, 0])
    lam_vals = np.unique(h_params[:, 1])

    # --- Figure 1: Residual landscape vs (T, λ) ---
    _plot_residual_landscape(cfg, h_params, frob_norms, coh_resids,
                             T_vals, lam_vals, h_t_fs, save_dir)

    # --- Figure 2: Residual time traces for 3 regimes ---
    _plot_residual_traces(cfg, h_params, h_rho, lstm_preds,
                          frob_norms, coh_resids, rc_resids,
                          h_t_fs, T_vals, lam_vals, save_dir)

    # --- Figure 3: ||R||_F vs T (quantum-classical crossover) ---
    _plot_nm_signature(cfg, h_params, frob_norms, h_t_fs, T_vals, lam_vals, save_dir)

    # --- Print summary ---
    print("\n=== Non-Markovian Residual Summary ===")
    for T in T_vals:
        mask = h_params[:, 0] == T
        mean_frob_end = frob_norms[mask, -1].mean()
        mean_coh_peak = np.abs(coh_resids[mask]).max(axis=1).mean()
        sign = "+" if coh_resids[mask, :n_steps//3].mean() > 0 else "-"
        print(f"  T={T:.0f}K: ||R||_F(t_end)={mean_frob_end:.4f}, "
              f"|R_coh|_peak={mean_coh_peak:.4f} (sign={sign})")
    print("  Positive R_coh → HEOM has MORE coherence (memory extends coherence)")
    print("  Negative R_coh → HEOM has LESS coherence (non-Markovian damping)")

    return {
        "frob_norms": frob_norms,
        "coh_resids": coh_resids,
        "rc_resids": rc_resids,
        "T_vals": T_vals,
        "lam_vals": lam_vals,
        "mean_frob_end": float(frob_norms[:, -1].mean()),
    }


def _plot_residual_landscape(cfg, h_params, frob_norms, coh_resids,
                              T_vals, lam_vals, t_fs, save_dir):
    """Heatmap of residual norm at t=200 fs and t_end vs (T, λ)."""
    set_plot_style()
    ev = cfg["evaluation"]

    # Time index closest to 200 fs
    t200_idx = int(np.argmin(np.abs(t_fs - 200.0)))
    t_end_idx = -1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, t_idx, t_label in [
        (axes[0], t200_idx, f"t = 200 fs"),
        (axes[1], t_end_idx, f"t = {t_fs[-1]:.0f} fs (t_end)"),
    ]:
        grid = np.zeros((len(T_vals), len(lam_vals)))
        for i, T in enumerate(T_vals):
            for j, lam in enumerate(lam_vals):
                mask = (h_params[:, 0] == T) & (h_params[:, 1] == lam)
                if mask.sum() > 0:
                    grid[i, j] = frob_norms[mask, t_idx].mean()

        T_mesh, L_mesh = np.meshgrid(T_vals, lam_vals, indexing="ij")
        im = ax.pcolormesh(T_mesh, L_mesh, grid, cmap="hot_r",
                           shading="auto", vmin=0)
        plt.colorbar(im, ax=ax, label=r"$\|R\|_F$ (Frobenius norm)")
        if grid.max() > grid.min() + 1e-6:
            cs = ax.contour(T_mesh, L_mesh, gaussian_filter(grid, 0.5),
                            levels=6, colors="navy", alpha=0.5, linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f")
        ax.set_xlabel("Temperature T (K)", fontsize=12)
        ax.set_ylabel(r"$\lambda$ (cm$^{-1}$)", fontsize=12)
        ax.set_title(f"Non-Markovian Residual ‖R‖_F at {t_label}\n"
                     r"$\|$HEOM$-$LSTM$\|_F$", fontsize=11)

    fig.suptitle("Non-Markovian Fingerprint: Lindblad–LSTM vs HEOM\n"
                 "Larger residual = stronger memory / vibronic effects",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, "residual_landscape", save_dir, ev["figure_dpi"])
    plt.close(fig)


def _plot_residual_traces(cfg, h_params, h_rho, lstm_preds,
                           frob_norms, coh_resids, rc_resids,
                           t_fs, T_vals, lam_vals, save_dir):
    """Time traces of coherence and RC residuals for 3 representative regimes."""
    set_plot_style()
    ev = cfg["evaluation"]

    # Select 3 representative (T, λ) points
    regimes = []
    for T, lam, label, color in [
        (77.0,  10.0,  "Low T, Low λ\n(quantum/ballistic)", "royalblue"),
        (200.0, 55.0,  "Mid T, Mid λ\n(ENAQT optimum)",     "forestgreen"),
        (300.0, 100.0, "High T, High λ\n(classical/Zeno)",  "crimson"),
    ]:
        mask = (h_params[:, 0] == T) & (h_params[:, 1] == lam)
        if mask.sum() == 0:
            # Fallback to nearest available
            dist = (h_params[:, 0] - T)**2 + (h_params[:, 1] - lam)**2
            mask = np.zeros(len(h_params), dtype=bool)
            mask[np.argmin(dist)] = True
        if mask.sum() > 0:
            idx = np.where(mask)[0][0]
            regimes.append((idx, label, color,
                            h_params[idx, 0], h_params[idx, 1]))

    fig, axes = plt.subplots(2, len(regimes), figsize=(5 * len(regimes), 9))
    if len(regimes) == 1:
        axes = axes.reshape(2, 1)

    for col, (idx, label, color, T_act, lam_act) in enumerate(regimes):
        heom_coh  = extract_coherence_12(h_rho[idx])
        lstm_coh  = extract_coherence_12(lstm_preds[idx])
        heom_rc   = extract_populations(h_rho[idx])[:, 6]
        lstm_rc   = extract_populations(lstm_preds[idx])[:, 6]

        # Top row: coherence + residual
        ax = axes[0, col]
        ax.plot(t_fs, heom_coh, "r-",  lw=2,   label="HEOM", alpha=0.9)
        ax.plot(t_fs, lstm_coh, "b--", lw=1.5, label="LSTM", alpha=0.9)
        ax2 = ax.twinx()
        ax2.fill_between(t_fs, coh_resids[idx], 0,
                         alpha=0.25, color="gray",
                         label=r"$R_{coh}$ = HEOM − LSTM")
        ax2.axhline(0, color="gray", lw=0.8, ls=":")
        ax2.set_ylabel(r"$R_{coh}$ (residual)", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax.set_title(f"{label}\nT={T_act:.0f}K, λ={lam_act:.0f} cm⁻¹", fontsize=10)
        ax.set_ylabel(r"$|\rho_{12}(t)|$ Coherence", fontsize=10)
        if col == 0:
            ax.legend(fontsize=9, loc="upper right")

        # Bottom row: RC population + residual
        ax = axes[1, col]
        ax.plot(t_fs, heom_rc, "r-",  lw=2,   label="HEOM", alpha=0.9)
        ax.plot(t_fs, lstm_rc, "b--", lw=1.5, label="LSTM", alpha=0.9)
        ax2 = ax.twinx()
        ax2.fill_between(t_fs, rc_resids[idx], 0,
                         alpha=0.25, color="purple",
                         label=r"$R_{RC}$ = HEOM − LSTM")
        ax2.axhline(0, color="purple", lw=0.8, ls=":")
        ax2.set_ylabel(r"$R_{RC}$ (residual)", fontsize=10, color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")
        ax.set_xlabel("Time (fs)", fontsize=10)
        ax.set_ylabel(r"$P_7(t)$ RC-proximal population", fontsize=10)
        if col == 0:
            ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Non-Markovian Residual: HEOM − LSTM(Lindblad)\n"
                 "Gray/purple shading = what Markovian approximation misses",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, "residual_trajectories", save_dir, ev["figure_dpi"])
    plt.close(fig)


def _plot_nm_signature(cfg, h_params, frob_norms, t_fs, T_vals, lam_vals, save_dir):
    """||R||_F vs T — shows quantum-classical crossover."""
    set_plot_style()
    ev = cfg["evaluation"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ||R||_F vs T, one curve per λ
    ax = axes[0]
    for lam in lam_vals:
        mean_frob = []
        for T in T_vals:
            mask = (h_params[:, 0] == T) & (h_params[:, 1] == lam)
            if mask.sum() > 0:
                mean_frob.append(frob_norms[mask].mean(axis=0).mean())
            else:
                mean_frob.append(np.nan)
        ax.plot(T_vals, mean_frob, "o-", lw=2, ms=7, label=f"λ={lam:.0f} cm⁻¹")
    ax.set_xlabel("Temperature T (K)", fontsize=12)
    ax.set_ylabel(r"Mean $\|R\|_F$ (time-averaged)", fontsize=12)
    ax.set_title("Non-Markovian Signature vs Temperature\n"
                 r"($\|$HEOM $-$ LSTM$\|_F$ averaged over all $t$ and $\lambda$)",
                 fontsize=11)
    ax.legend(fontsize=10)

    # Right: Time profile of ||R||_F averaged over all parameter combos
    ax = axes[1]
    for T in T_vals:
        mask = h_params[:, 0] == T
        if mask.sum() > 0:
            mean_profile = frob_norms[mask].mean(axis=0)
            ax.plot(t_fs, mean_profile, lw=2, label=f"T={T:.0f}K")
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel(r"$\|R(t)\|_F$ (λ-averaged)", fontsize=12)
    ax.set_title("Time Profile of Non-Markovian Residual\n"
                 "Peak at early times → memory effect timescale",
                 fontsize=11)
    ax.legend(fontsize=10)

    fig.suptitle("Quantum–Classical Crossover in Non-Markovian Effects\n"
                 "HEOM − Lindblad residual characterises bath memory",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, "non_markovian_signature", save_dir, ev["figure_dpi"])
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_non_markovian_analysis(cfg, device)
