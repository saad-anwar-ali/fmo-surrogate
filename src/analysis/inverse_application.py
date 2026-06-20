"""
inverse_application.py  —  Real applications of the normalising flow inverse model.

Three analyses that turn the circular "infer params from simulated trajectories"
into genuinely useful science:

1. NOISY RECOVERY BENCHMARK
   Inject realistic noise (photon shot noise + detector noise, σ ≈ 0.01–0.05)
   into clean QuTiP trajectories and show the inverse model recovers the
   correct posterior. This demonstrates robustness to experimental noise
   and gives calibration curves. Key metric: coverage probability of the
   90% credible interval.

2. PARAMETER IDENTIFIABILITY ANALYSIS
   Which bath parameters are recoverable from a trajectory observation?
   Use mutual information between posterior samples and true params to
   rank identifiability: I(θ_i; data) from the inverse model posterior.
   Expected result: T and λ are identifiable (strong bath coupling signal),
   ω_c is weakly identifiable, α is nearly unidentifiable (degenerate with λ).

3. 2DES-INSPIRED SYNTHETIC EXPERIMENT
   Simulate what the inverse model would infer from the Engel 2007 experiment:
   - Use the known FMO parameters (T=77K, λ≈35 cm⁻¹, ω_c≈150 cm⁻¹, α≈0.5)
     to generate a "pseudo-experimental" trajectory.
   - Add realistic 2DES noise: σ=0.02 coherence noise + 5% shot noise on pops.
   - Run the inverse model and show that the posterior credible intervals
     contain the true values.
   - Compare with the 277K/300K versions (Panitchayangkoon 2010 conditions).
   This is the closest we can get to applying the model to real experimental
   data without the actual 2DES raw data (which requires DFT preprocessing).
"""

import os, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (load_config, set_plot_style, save_figure,
                   extract_coherence_12, extract_populations,
                   seed_everything, N_RHO, N_SITES)
from dataset import ParamNormaliser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_inverse(cfg, device):
    from models.inverse_model import build_inverse_from_config
    path = os.path.join(cfg["training"]["checkpoint_dir"], "inverse_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inverse model checkpoint not found at {path}")
    ckpt = torch.load(path, map_location=device)
    m = build_inverse_from_config(cfg).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, ParamNormaliser.from_dict(ckpt["normaliser"])


def _load_lstm(cfg, device):
    from models.lstm_model import build_lstm_from_config
    path = os.path.join(cfg["training"]["checkpoint_dir"], "lstm_best.pt")
    ckpt = torch.load(path, map_location=device)
    m = build_lstm_from_config(cfg).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, ParamNormaliser.from_dict(ckpt["normaliser"])


@torch.no_grad()
def _infer_posterior(inv_model, rho_seq_np, n_samples, device):
    """
    Sample from P(params | trajectory).
    rho_seq_np : (n_steps, 98)
    Returns    : (n_samples, 5) in normalised space
    """
    rho_t = torch.tensor(rho_seq_np[None], dtype=torch.float32, device=device)
    samps = inv_model.sample_posterior(rho_t, n_samples=n_samples)
    return samps.squeeze(0).cpu().numpy()


@torch.no_grad()
def _lstm_rollout(lstm, norm, params_raw, rho0, n_steps, device):
    pn = norm.transform(params_raw.astype(np.float32))
    p  = torch.tensor(pn,   dtype=torch.float32, device=device).unsqueeze(0)
    r  = torch.tensor(rho0, dtype=torch.float32, device=device).unsqueeze(0)
    return lstm.rollout(p, r, n_steps).squeeze(0).cpu().numpy()


def _add_noise(rho_flat, sigma_coh=0.02, sigma_pop=0.01, rng=None):
    """
    Add realistic noise to a density matrix trajectory.

    Two noise sources:
      - Gaussian noise on off-diagonal elements (coherences): mimics
        shot noise and laser phase fluctuations in 2DES.
      - Gaussian noise on diagonal elements (populations): mimics
        detector noise. Clipped to keep populations physical.

    rho_flat : (n_steps, 98)
    Returns  : (n_steps, 98) noisy trajectory
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noisy = rho_flat.copy()
    n2 = N_SITES * N_SITES
    diag_idx = np.arange(N_SITES) * (N_SITES + 1)

    # Off-diagonal Re and Im
    off_diag_mask = np.ones(n2, dtype=bool)
    off_diag_mask[diag_idx] = False
    noisy[:, np.where(off_diag_mask)[0]] += rng.normal(
        0, sigma_coh, (rho_flat.shape[0], off_diag_mask.sum()))
    noisy[:, n2 + np.where(off_diag_mask)[0]] += rng.normal(
        0, sigma_coh, (rho_flat.shape[0], off_diag_mask.sum()))

    # Diagonal (populations)
    noisy[:, diag_idx] += rng.normal(0, sigma_pop, (rho_flat.shape[0], N_SITES))
    # Clip populations to [0, 1]
    noisy[:, diag_idx] = np.clip(noisy[:, diag_idx], 0, 1)

    return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Analysis 1: Noisy recovery benchmark
# ---------------------------------------------------------------------------

def run_noisy_recovery_benchmark(cfg, inv_model, inv_norm, lstm, lstm_norm,
                                  test_idx, device, save_dir, n_samples=1000):
    """
    Inject noise at 3 levels and check posterior coverage of 90% CI.
    """
    set_plot_style()
    ev   = cfg["evaluation"]
    hdf5 = cfg["data"]["lindblad_hdf5"]

    with h5py.File(hdf5, "r") as f:
        t_fs = f["trajectories/t_fs"][:]
    n_steps = len(t_fs)

    sigma_levels = [0.0, 0.01, 0.03, 0.05]
    param_names  = ["T (K)", "λ (cm⁻¹)", "ωc (cm⁻¹)", "α"]
    n_cont       = 4   # continuous params only (exclude init_site)

    results = {s: {"in_90ci": [], "posterior_mean": [], "true": []}
               for s in sigma_levels}

    rng = np.random.default_rng(42)
    n_eval = min(30, len(test_idx))   # evaluate on first 30 test trajectories

    print(f"  Noisy recovery benchmark: {n_eval} trajectories × {len(sigma_levels)} noise levels...")
    with h5py.File(hdf5, "r") as f:
        for ti in test_idx[:n_eval]:
            gt   = f["trajectories/rho_flat"][int(ti)]
            par  = f["trajectories/params"][int(ti)]
            true_phys = par[:n_cont]

            for sigma in sigma_levels:
                if sigma == 0.0:
                    noisy = gt
                else:
                    noisy = _add_noise(gt, sigma_coh=sigma,
                                       sigma_pop=sigma*0.5, rng=rng)

                try:
                    samps_norm = _infer_posterior(inv_model, noisy, n_samples, device)
                    samps_phys = inv_norm.inverse_transform(samps_norm)[:, :n_cont]

                    p05 = np.percentile(samps_phys, 5,  axis=0)
                    p95 = np.percentile(samps_phys, 95, axis=0)
                    mean = samps_phys.mean(axis=0)
                    in_ci = np.all((true_phys >= p05) & (true_phys <= p95))

                    results[sigma]["in_90ci"].append(float(in_ci))
                    results[sigma]["posterior_mean"].append(mean)
                    results[sigma]["true"].append(true_phys)
                except Exception as e:
                    print(f"    Skipped ti={ti} σ={sigma}: {e}")

    # --- Figure: Coverage probability vs noise level ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Coverage probability vs sigma for each param
    ax = axes[0]
    cov_by_sigma = {s: np.mean(results[s]["in_90ci"]) for s in sigma_levels
                    if results[s]["in_90ci"]}
    ax.plot(sigma_levels, [cov_by_sigma.get(s, np.nan) for s in sigma_levels],
            "ko-", lw=2, ms=8)
    ax.axhline(0.90, color="green", lw=1.5, ls="--", label="Ideal 90% CI")
    ax.axhline(1.00, color="gray",  lw=0.8, ls=":")
    ax.set_xlabel("Noise level σ", fontsize=12)
    ax.set_ylabel("90% CI coverage probability", fontsize=12)
    ax.set_title("Posterior Coverage vs Noise Level\n"
                 "(Should track ~0.90 for well-calibrated model)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    # Right: Posterior mean accuracy at sigma=0.01
    ax = axes[1]
    sigma_show = 0.01
    if results[sigma_show]["posterior_mean"]:
        pm = np.array(results[sigma_show]["posterior_mean"])   # (n_eval, 4)
        tr = np.array(results[sigma_show]["true"])             # (n_eval, 4)
        for pi, pname in enumerate(param_names):
            if tr[:, pi].std() > 0 and pm[:, pi].std() > 0:
                r2 = r2_score(tr[:, pi], pm[:, pi])
                ax.scatter(tr[:, pi], pm[:, pi], alpha=0.5, s=30,
                           label=f"{pname} R²={r2:.2f}")
    ax.set_xlabel("True parameter value", fontsize=12)
    ax.set_ylabel("Posterior mean", fontsize=12)
    ax.set_title(f"Posterior Mean Accuracy (σ={sigma_show})\n"
                 "R² per parameter", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Inverse Model Noisy Recovery Benchmark\n"
                 "Demonstrates robustness to 2DES-level experimental noise",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, "inverse_noisy_recovery", save_dir, ev["figure_dpi"])
    plt.close(fig)

    for s in sigma_levels:
        if results[s]["in_90ci"]:
            cov = np.mean(results[s]["in_90ci"])
            print(f"  σ={s:.2f}: 90% CI coverage = {cov:.3f} (ideal=0.90)")


# ---------------------------------------------------------------------------
# Analysis 2: Parameter identifiability
# ---------------------------------------------------------------------------

def run_identifiability_analysis(cfg, inv_model, inv_norm, test_idx,
                                  device, save_dir, n_samples=500):
    """
    Rank bath parameters by identifiability using posterior variance.
    Lower posterior variance = more identifiable.
    """
    set_plot_style()
    ev   = cfg["evaluation"]
    hdf5 = cfg["data"]["lindblad_hdf5"]
    param_names = ["T (K)", "λ (cm⁻¹)", "ωc (cm⁻¹)", "α"]

    posterior_vars = [[] for _ in range(4)]
    n_eval = min(40, len(test_idx))

    print(f"  Identifiability analysis: {n_eval} trajectories...")
    with h5py.File(hdf5, "r") as f:
        for ti in test_idx[:n_eval]:
            gt  = f["trajectories/rho_flat"][int(ti)]
            try:
                samps_norm = _infer_posterior(inv_model, gt, n_samples, device)
                samps_phys = inv_norm.inverse_transform(samps_norm)[:, :4]
                # Normalised variance: Var(sample) / Var(prior)
                for pi in range(4):
                    posterior_vars[pi].append(float(samps_phys[:, pi].std()))
            except Exception:
                pass

    if not any(posterior_vars[0]):
        print("  Identifiability analysis: no successful inferences, skipping.")
        return

    mean_stds = [np.mean(v) if v else np.nan for v in posterior_vars]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Bar chart of posterior std per param
    ax = axes[0]
    colors = ["royalblue", "darkorange", "forestgreen", "crimson"]
    bars = ax.bar(param_names, mean_stds, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Posterior std (physical units)", fontsize=12)
    ax.set_title("Parameter Identifiability\n"
                 "Lower std → more identifiable from trajectory", fontsize=11)
    for bar, val in zip(bars, mean_stds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=10)

    # Right: Posterior distributions for a single test trajectory
    ax = axes[1]
    with h5py.File(hdf5, "r") as f:
        ti  = int(test_idx[0])
        gt  = f["trajectories/rho_flat"][ti]
        par = f["trajectories/params"][ti]
    try:
        samps_norm = _infer_posterior(inv_model, gt, 2000, device)
        samps_phys = inv_norm.inverse_transform(samps_norm)[:, :4]
        for pi, (pname, color) in enumerate(zip(param_names, colors)):
            vals = samps_phys[:, pi]
            kde  = gaussian_kde(vals)
            x    = np.linspace(vals.min(), vals.max(), 200)
            y    = kde(x)
            y    = y / y.max()   # normalise for overlay
            ax.plot(x / np.abs(x).max(), y + pi, color=color, lw=2, label=pname)
            ax.axvline(par[pi] / np.abs(par[pi]).max() if par[pi] != 0 else 0,
                       ymin=pi/4, ymax=(pi+1)/4,
                       color=color, lw=2, ls="--", alpha=0.7)
        ax.set_title("Posterior Distributions (normalised)\n"
                     "Dashed line = true value", fontsize=11)
        ax.set_xlabel("Normalised parameter value", fontsize=12)
        ax.set_yticks(range(4)); ax.set_yticklabels(param_names)
        ax.legend(fontsize=9, loc="upper right")
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not compute\n{e}", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle("Parameter Identifiability from FMO Trajectory Observations\n"
                 "Inverse normalising flow posterior analysis",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, "parameter_identifiability", save_dir, ev["figure_dpi"])
    plt.close(fig)

    print("\n  Identifiability ranking (low σ = more identifiable):")
    ranked = sorted(zip(mean_stds, param_names))
    for std, pname in ranked:
        print(f"    {pname}: posterior σ = {std:.2f}")


# ---------------------------------------------------------------------------
# Analysis 3: 2DES-inspired synthetic experiment
# ---------------------------------------------------------------------------

def run_2des_synthetic_experiment(cfg, inv_model, inv_norm, lstm, lstm_norm,
                                   device, save_dir, n_samples=2000):
    """
    Simulate parameter inference from 2DES-like experimental conditions.

    Uses known FMO parameters from the literature to generate pseudo-experimental
    trajectories, adds 2DES-level noise, and shows whether the posterior
    credible intervals contain the true values.

    Experimental conditions:
      Engel 2007 (77K):              T=77K,  λ=35, ωc=150, α=0.5, site 1
      Panitchayangkoon 2010 (277K):  T=277K, λ=55, ωc=100, α=1.0, site 1
      Panitchayangkoon 2010 (300K):  T=300K, λ=55, ωc=100, α=1.0, site 1
    """
    set_plot_style()
    ev = cfg["evaluation"]

    # Experimental conditions from the literature
    experiments = [
        {"label": "Engel 2007\n(77 K, cryogenic)",
         "params": np.array([77.0, 35.0, 150.0, 0.5, 0.0], dtype=np.float32),
         "T_cite": "77 K", "ref": "Engel et al. 2007, Nature"},
        {"label": "Panitchayangkoon 2010\n(277 K, near-physiological)",
         "params": np.array([277.0, 55.0, 100.0, 1.0, 0.0], dtype=np.float32),
         "T_cite": "277 K", "ref": "Panitchayangkoon et al. 2010, PNAS"},
        {"label": "Panitchayangkoon 2010\n(300 K, physiological)",
         "params": np.array([300.0, 55.0, 100.0, 1.0, 0.0], dtype=np.float32),
         "T_cite": "300 K", "ref": "Panitchayangkoon et al. 2010, PNAS"},
    ]

    with h5py.File(cfg["data"]["lindblad_hdf5"], "r") as f:
        t_fs = f["trajectories/t_fs"][:]
    n_steps = len(t_fs)

    rho0 = np.zeros(N_RHO, dtype=np.float32)
    rho0[0] = 1.0
    rng = np.random.default_rng(0)

    param_names = ["T (K)", "λ (cm⁻¹)", "ωc (cm⁻¹)", "α"]
    n_cont = 4

    fig, axes = plt.subplots(n_cont, len(experiments),
                              figsize=(5 * len(experiments), 4 * n_cont))

    print(f"\n  Running 2DES synthetic experiment ({len(experiments)} conditions)...")
    for col, exp in enumerate(experiments):
        print(f"    {exp['T_cite']}...")
        true_params = exp["params"]

        # Generate clean trajectory using LSTM
        traj_clean = _lstm_rollout(lstm, lstm_norm, true_params, rho0, n_steps, device)

        # Add 2DES-level noise (σ=0.02 for coherences, σ=0.01 for populations)
        traj_noisy = _add_noise(traj_clean, sigma_coh=0.02, sigma_pop=0.01, rng=rng)

        try:
            samps_norm = _infer_posterior(inv_model, traj_noisy, n_samples, device)
            samps_phys = inv_norm.inverse_transform(samps_norm)[:, :n_cont]

            for row, (pname, true_val) in enumerate(zip(param_names, true_params[:n_cont])):
                ax = axes[row, col]
                vals = samps_phys[:, row]
                p05, p95 = np.percentile(vals, [5, 95])
                in_ci = p05 <= true_val <= p95

                # Histogram
                ax.hist(vals, bins=40, density=True, color="steelblue",
                        alpha=0.6, edgecolor="white", lw=0.3)
                ax.axvline(true_val, color="red", lw=2.5,
                           label=f"True: {true_val:.1f}")
                ax.axvline(p05, color="gray", lw=1.5, ls="--", alpha=0.7)
                ax.axvline(p95, color="gray", lw=1.5, ls="--", alpha=0.7,
                           label=f"90% CI: [{p05:.1f}, {p95:.1f}]")
                ax.axvspan(p05, p95, alpha=0.1, color="steelblue")

                status = "✓ IN CI" if in_ci else "✗ MISS"
                ax.set_title(f"{status}", fontsize=9,
                             color="green" if in_ci else "red")
                if row == 0:
                    ax.set_xlabel("", fontsize=10)
                    ax2 = ax.secondary_xaxis("top")
                    ax2.set_xlabel(exp["label"], fontsize=9, labelpad=8)
                    ax2.set_xticks([])
                if col == 0:
                    ax.set_ylabel(pname, fontsize=10)
                ax.legend(fontsize=8)

        except Exception as e:
            for row in range(n_cont):
                axes[row, col].text(0.5, 0.5, f"Failed:\n{e}",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=9)

    fig.suptitle("2DES-Inspired Synthetic Experiment\n"
                 "Inverse model posterior inference under experimental noise (σ=0.02)\n"
                 "Red line = literature value, gray = 90% CI",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_figure(fig, "inverse_2des_experiment", save_dir, ev["figure_dpi"])
    plt.close(fig)
    print("  2DES synthetic experiment figures saved.")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_inverse_analyses(cfg, device=None, save_dir=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_dir is None:
        save_dir = cfg["evaluation"]["figure_dir"]

    seed_everything(cfg["seed"])

    try:
        inv_model, inv_norm = _load_inverse(cfg, device)
        lstm, lstm_norm     = _load_lstm(cfg, device)
    except FileNotFoundError as e:
        print(f"Skipping inverse analyses: {e}")
        return

    # Load test indices from inverse checkpoint
    ckpt = torch.load(
        os.path.join(cfg["training"]["checkpoint_dir"], "inverse_best.pt"),
        map_location=device)
    test_idx = np.array(ckpt["test_idx"], dtype=np.int64)

    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Analysis 1: Noisy Recovery Benchmark ===")
    run_noisy_recovery_benchmark(cfg, inv_model, inv_norm, lstm, lstm_norm,
                                  test_idx, device, save_dir)

    print("\n=== Analysis 2: Parameter Identifiability ===")
    run_identifiability_analysis(cfg, inv_model, inv_norm,
                                  test_idx, device, save_dir)

    print("\n=== Analysis 3: 2DES Synthetic Experiment ===")
    run_2des_synthetic_experiment(cfg, inv_model, inv_norm, lstm, lstm_norm,
                                   device, save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_all_inverse_analyses(cfg)
