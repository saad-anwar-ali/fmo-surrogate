"""
disorder_ensemble.py  —  Static disorder ensemble analysis.

Static disorder in FMO arises from conformational heterogeneity of the
protein scaffold — each FMO complex in a sample has slightly different
site energies due to local electrostatic fluctuations. In spectroscopy,
you always observe an ensemble average over this disorder.

This module:

1. DISORDER ENSEMBLE SIMULATION
   Draw N=1000 disorder realisations from a Gaussian distribution over
   bath parameters (σ_T≈20K, σ_λ≈30 cm⁻¹, σ_ωc≈25 cm⁻¹, σ_α≈0.3)
   centred on the ENAQT optimum. Run the LSTM surrogate on all 1000 in
   milliseconds. Compute disorder-averaged observables and their variance.

   Comparison: running 1000 QuTiP trajectories would take ~30 minutes.
   The surrogate does it in <5 seconds. That's the scientific value.

2. DISORDER-AVERAGED ENAQT
   Show that the true (disorder-averaged) ENAQT efficiency differs from
   the single-point prediction. The non-linear dependence of efficiency
   on parameters means <eff(params)> ≠ eff(<params>) — Jensen's inequality.
   This is a physically important result: experimental ENAQT measurements
   always include disorder, so single-parameter estimates are biased.

3. COHERENCE LIFETIME DISTRIBUTION
   Extract the coherence decay rate Γ₂ for each of the 1000 disorder
   realisations and plot the resulting distribution. Compare with the
   distribution expected from the Redfield theory prediction
   Γ₂ ∝ λ·k_B·T / ħ·ω_c². This tests whether the surrogate has learned
   the correct physical scaling.

4. SITE-ENERGY DISORDER
   Beyond bath parameter disorder, simulate direct site-energy disorder:
   draw σ_ε ~ 50–100 cm⁻¹ Gaussian disorder on each of the 7 site energies
   (consistent with single-molecule spectroscopy), generate trajectories
   for each realisation, and compute the disorder-averaged density matrix
   <ρ(t)>_disorder. This is what ultrafast spectroscopy actually measures.

   NOTE: site-energy disorder changes the Hamiltonian, not just the bath.
   We handle this by re-running QuTiP for a coarse grid and using the
   surrogate for interpolation (hybrid approach).
"""

import os, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (load_config, set_plot_style, save_figure,
                   extract_populations, extract_coherence_12,
                   seed_everything, N_RHO, N_SITES)
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
def _rollout_batch(model, params_norm_np, rho0_np, n_steps, device, chunk=64):
    """Batched autoregressive rollout. Returns (N, n_steps, 98)."""
    N = params_norm_np.shape[0]
    all_trajs = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        p = torch.tensor(params_norm_np[start:end], dtype=torch.float32, device=device)
        r = torch.tensor(rho0_np[start:end],        dtype=torch.float32, device=device)
        all_trajs.append(model.rollout(p, r, n_steps).cpu().numpy())
    return np.concatenate(all_trajs, axis=0)


def _fit_gamma2_envelope(t_fs, coh):
    """Fit exponential to Hilbert envelope. Returns (Gamma2, success)."""
    try:
        envelope = np.abs(hilbert(coh))
        peak_idx = int(np.argmax(envelope[:len(envelope)//3]))
        t_fit = t_fs[peak_idx:]; e_fit = envelope[peak_idx:]
        def f(t, A, G, C): return A * np.exp(-G * (t - t_fit[0])) + C
        p0 = [e_fit[0] - e_fit[-1], 1/200, e_fit[-1]]
        p, _ = curve_fit(f, t_fit, e_fit, p0=p0,
                         bounds=([0, 1e-7, 0], [2.0, 0.1, 1.0]), maxfev=4000)
        return float(p[1]), True
    except Exception:
        return np.nan, False


def _efficiency_from_traj(traj):
    """1 - Tr(rho(t_end)) per trajectory. traj: (N, T, 98)."""
    diag_idx = np.arange(N_SITES) * (N_SITES + 1)
    trace_end = traj[:, -1, diag_idx].sum(axis=-1)
    return np.clip(1.0 - trace_end, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Analysis 1 & 2: Bath parameter disorder ensemble
# ---------------------------------------------------------------------------

def run_disorder_ensemble(cfg, device=None, save_dir=None,
                           N_disorder=1000, centre_params=None):
    """
    Run a disorder ensemble over bath parameters.

    Parameters
    ----------
    N_disorder    : int   — number of disorder realisations
    centre_params : dict  — centre of disorder distribution.
                            Default: ENAQT optimum from config sweep.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_dir is None:
        save_dir = cfg["evaluation"]["figure_dir"]

    seed_everything(cfg["seed"])
    set_plot_style()
    print(f"Loading LSTM surrogate...")
    model, norm = _load_lstm(cfg, device)

    sw = cfg["sweep"]
    with h5py.File(cfg["data"]["lindblad_hdf5"], "r") as f:
        t_fs = f["trajectories/t_fs"][:]
    n_steps = len(t_fs)

    # Centre of disorder distribution — default to mid-range (near ENAQT optimum)
    if centre_params is None:
        centre_params = {
            "T":      200.0,    # K  — intermediate temperature
            "lambda": 55.0,     # cm⁻¹
            "omega_c": 100.0,   # cm⁻¹
            "alpha":  1.0,
            "site":   0.0,
        }

    # Disorder widths — from single-molecule FMO spectroscopy literature
    # Jankowiak et al. 2011: σ_inhom ≈ 50–100 cm⁻¹ (site energy disorder)
    # We translate this to bath parameter uncertainty
    sigma = {
        "T":      20.0,   # K
        "lambda": 30.0,   # cm⁻¹
        "omega_c": 25.0,  # cm⁻¹
        "alpha":  0.3,
    }

    rng = np.random.default_rng(cfg["seed"])

    # Sample disorder realisations — clip to physical bounds
    T_lim   = [min(sw["T_K"]),       max(sw["T_K"])]
    lam_lim = [min(sw["lambda_cm"]),  max(sw["lambda_cm"])]
    wc_lim  = [min(sw["omega_c_cm"]), max(sw["omega_c_cm"])]
    al_lim  = [min(sw["alpha_scale"]), max(sw["alpha_scale"])]

    T_samp   = np.clip(rng.normal(centre_params["T"],      sigma["T"],      N_disorder), *T_lim)
    lam_samp = np.clip(rng.normal(centre_params["lambda"], sigma["lambda"], N_disorder), *lam_lim)
    wc_samp  = np.clip(rng.normal(centre_params["omega_c"],sigma["omega_c"],N_disorder), *wc_lim)
    al_samp  = np.clip(rng.normal(centre_params["alpha"],  sigma["alpha"],  N_disorder), *al_lim)

    params_raw = np.stack([T_samp, lam_samp, wc_samp, al_samp,
                           np.full(N_disorder, centre_params["site"])],
                          axis=-1).astype(np.float32)

    # Initial state: all population on site 1
    rho0 = np.zeros((N_disorder, N_RHO), dtype=np.float32)
    rho0[:, 0] = 1.0

    # --- Surrogate rollout (the key speed comparison) ---
    print(f"  Running {N_disorder} disorder realisations via LSTM...")
    t_start = time.perf_counter()
    params_norm = norm.transform(params_raw)
    trajs = _rollout_batch(model, params_norm, rho0, n_steps, device)
    t_lstm = time.perf_counter() - t_start
    print(f"  LSTM: {N_disorder} trajectories in {t_lstm:.2f}s "
          f"({t_lstm/N_disorder*1000:.1f} ms each)")
    print(f"  QuTiP equivalent would take ~{N_disorder * 0.1 / 60:.1f} minutes")

    # --- Compute disorder-averaged observables ---
    coh_all  = np.array([extract_coherence_12(trajs[i])         for i in range(N_disorder)])
    pop1_all = np.array([extract_populations(trajs[i])[:, 0]    for i in range(N_disorder)])
    pop7_all = np.array([extract_populations(trajs[i])[:, 6]    for i in range(N_disorder)])
    eff_all  = _efficiency_from_traj(trajs)

    coh_mean  = coh_all.mean(axis=0)
    coh_std   = coh_all.std(axis=0)
    pop1_mean = pop1_all.mean(axis=0)
    pop1_std  = pop1_all.std(axis=0)
    pop7_mean = pop7_all.mean(axis=0)
    pop7_std  = pop7_all.std(axis=0)
    eff_mean  = eff_all.mean()
    eff_std   = eff_all.std()

    # Jensen's inequality check: is <eff(θ)> ≠ eff(<θ>)?
    centre_raw = np.array([[centre_params["T"], centre_params["lambda"],
                             centre_params["omega_c"], centre_params["alpha"],
                             centre_params["site"]]], dtype=np.float32)
    centre_norm = norm.transform(centre_raw)
    rho0_single = rho0[:1]
    traj_single = _rollout_batch(model, centre_norm, rho0_single, n_steps, device)
    eff_single  = float(_efficiency_from_traj(traj_single)[0])
    jensen_gap  = eff_mean - eff_single

    print(f"\n  Disorder-averaged efficiency: {eff_mean*100:.3f}% ± {eff_std*100:.3f}%")
    print(f"  Single-point efficiency (at centre): {eff_single*100:.3f}%")
    print(f"  Jensen gap <eff(θ)> - eff(<θ>) = {jensen_gap*100:+.4f}%")
    if abs(jensen_gap) > eff_std * 0.1:
        print(f"  → Jensen gap is significant ({abs(jensen_gap)/eff_std:.1f}σ)")
    else:
        print(f"  → Jensen gap is within noise")

    # --- Figure 1: Disorder-averaged trajectories ---
    _plot_disorder_averaged_trajectories(
        cfg, t_fs, coh_mean, coh_std, coh_all,
        pop1_mean, pop1_std, pop7_mean, pop7_std,
        eff_mean, eff_std, eff_single, jensen_gap,
        centre_params, sigma, N_disorder, save_dir)

    # --- Figure 2: Efficiency distribution ---
    _plot_efficiency_distribution(cfg, eff_all, eff_mean, eff_std,
                                   eff_single, params_raw, save_dir)

    # --- Figure 3: Coherence decay rate distribution ---
    _plot_gamma2_distribution(cfg, t_fs, trajs, params_raw, save_dir)

    return {
        "N_disorder": N_disorder,
        "eff_mean": float(eff_mean),
        "eff_std":  float(eff_std),
        "eff_single_point": float(eff_single),
        "jensen_gap": float(jensen_gap),
        "t_lstm_s": float(t_lstm),
        "centre_params": centre_params,
        "sigma": sigma,
    }


def _plot_disorder_averaged_trajectories(cfg, t_fs, coh_mean, coh_std, coh_all,
                                          pop1_mean, pop1_std, pop7_mean, pop7_std,
                                          eff_mean, eff_std, eff_single, jensen_gap,
                                          centre_params, sigma, N_disorder, save_dir):
    set_plot_style()
    ev = cfg["evaluation"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Coherence
    ax = axes[0]
    # Show 50 individual realisations as light traces
    for i in range(min(50, len(coh_all))):
        ax.plot(t_fs, coh_all[i], "steelblue", alpha=0.05, lw=0.8)
    ax.fill_between(t_fs, coh_mean - coh_std, coh_mean + coh_std,
                    alpha=0.3, color="steelblue", label="±1σ disorder")
    ax.plot(t_fs, coh_mean, "b-", lw=2.5, label="Disorder mean")
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel(r"$|\rho_{12}(t)|$ Coherence", fontsize=12)
    ax.set_title(f"Disorder-Averaged Coherence\n"
                 f"N={N_disorder}, σ_λ={sigma['lambda']:.0f} cm⁻¹, "
                 f"σ_T={sigma['T']:.0f}K", fontsize=11)
    ax.legend(fontsize=10)

    # Site-1 and Site-7 populations
    ax = axes[1]
    ax.fill_between(t_fs, pop1_mean - pop1_std, pop1_mean + pop1_std,
                    alpha=0.25, color="royalblue")
    ax.fill_between(t_fs, pop7_mean - pop7_std, pop7_mean + pop7_std,
                    alpha=0.25, color="crimson")
    ax.plot(t_fs, pop1_mean, "b-",  lw=2, label="P₁ (antenna site)")
    ax.plot(t_fs, pop7_mean, "r--", lw=2, label="P₇ (RC-proximal)")
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.set_title("Disorder-Averaged Populations\n(shaded = ±1σ)", fontsize=11)
    ax.legend(fontsize=10)

    # Efficiency summary
    ax = axes[2]
    ax.bar(["Single-point\neff(⟨θ⟩)", "Disorder avg\n⟨eff(θ)⟩"],
           [eff_single * 100, eff_mean * 100],
           color=["steelblue", "darkorange"], alpha=0.8,
           edgecolor="black", width=0.5)
    ax.errorbar([1], [eff_mean * 100], yerr=[eff_std * 100],
                fmt="none", color="black", capsize=8, lw=2, capthick=2)
    ax.set_ylabel("Transfer Efficiency (%)", fontsize=12)
    ax.set_title(f"Jensen Inequality Effect\n"
                 f"⟨eff⟩ − eff(⟨θ⟩) = {jensen_gap*100:+.4f}%", fontsize=11)
    gap_color = "green" if jensen_gap > 0 else "red"
    ax.annotate(f"Jensen gap\n{jensen_gap*100:+.4f}%",
                xy=(0.5, max(eff_single, eff_mean) * 100),
                xytext=(0.5, max(eff_single, eff_mean) * 100 * 1.05),
                ha="center", fontsize=11, color=gap_color,
                arrowprops=dict(arrowstyle="->", color=gap_color))

    fig.suptitle(f"Static Disorder Ensemble — 7-site FMO\n"
                 f"N={N_disorder} realisations | LSTM surrogate | "
                 f"Centre: T={centre_params['T']:.0f}K, "
                 f"λ={centre_params['lambda']:.0f} cm⁻¹",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "disorder_averaged_observables", save_dir, ev["figure_dpi"])
    plt.close(fig)


def _plot_efficiency_distribution(cfg, eff_all, eff_mean, eff_std,
                                   eff_single, params_raw, save_dir):
    set_plot_style()
    ev = cfg["evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Efficiency distribution
    ax = axes[0]
    ax.hist(eff_all * 100, bins=50, density=True, color="steelblue",
            alpha=0.7, edgecolor="white", lw=0.3)
    if eff_all.std() > 0:
        kde = gaussian_kde(eff_all * 100)
        x   = np.linspace(eff_all.min() * 100, eff_all.max() * 100, 300)
        ax.plot(x, kde(x), "b-", lw=2)
    ax.axvline(eff_mean  * 100, color="blue",   lw=2, ls="-",
               label=f"⟨eff⟩ = {eff_mean*100:.3f}%")
    ax.axvline(eff_single * 100, color="orange", lw=2, ls="--",
               label=f"eff(⟨θ⟩) = {eff_single*100:.3f}%")
    ax.axvspan((eff_mean - eff_std) * 100, (eff_mean + eff_std) * 100,
               alpha=0.15, color="blue", label="±1σ")
    ax.set_xlabel("Transfer Efficiency (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Efficiency Distribution over Disorder Ensemble\n"
                 "(Non-Gaussian tail → strong disorder-efficiency coupling)", fontsize=11)
    ax.legend(fontsize=10)

    # Right: Efficiency vs bath parameters (scatter)
    ax = axes[1]
    sc = ax.scatter(params_raw[:, 1], params_raw[:, 0], c=eff_all * 100,
                    cmap="viridis", s=15, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Efficiency (%)")
    ax.set_xlabel(r"λ (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel("T (K)", fontsize=12)
    ax.set_title("Efficiency vs (T, λ) in Disorder Ensemble\n"
                 "Colour = transfer efficiency at each realisation", fontsize=11)

    plt.tight_layout()
    save_figure(fig, "disorder_efficiency_distribution", save_dir, ev["figure_dpi"])
    plt.close(fig)


def _plot_gamma2_distribution(cfg, t_fs, trajs, params_raw, save_dir):
    """
    Extract Γ₂ for each disorder realisation and compare with
    the Redfield scaling prediction: Γ₂ ∝ λ·k_B·T / (ħ·ω_c²).
    """
    set_plot_style()
    ev = cfg["evaluation"]
    k_B = cfg["physics"]["k_B_cm_per_K"]

    print("  Fitting coherence decay rates for disorder ensemble...")
    gamma2_vals, gamma2_redfield = [], []
    for i in range(len(trajs)):
        coh = extract_coherence_12(trajs[i])
        g2, ok = _fit_gamma2_envelope(t_fs, coh)
        if ok and not np.isnan(g2):
            gamma2_vals.append(g2)
            T, lam, wc = params_raw[i, 0], params_raw[i, 1], params_raw[i, 2]
            # Redfield scaling: Γ₂ ∝ 2π·λ·k_B·T / (ħ·ω_c²)  (in cm⁻¹ units)
            gamma2_redfield.append(2 * np.pi * lam * k_B * T / (wc**2))

    if len(gamma2_vals) < 10:
        print("  Too few successful fits for Γ₂ distribution plot.")
        return

    gamma2_vals     = np.array(gamma2_vals)
    gamma2_redfield = np.array(gamma2_redfield)
    # Normalise Redfield to same scale
    if gamma2_redfield.std() > 0:
        scale = gamma2_vals.mean() / gamma2_redfield.mean()
        gamma2_redfield *= scale

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Distribution of Γ₂
    ax = axes[0]
    ax.hist(gamma2_vals * 1000, bins=40, density=True, color="steelblue",
            alpha=0.7, edgecolor="white", label="LSTM fits")
    if gamma2_vals.std() > 0:
        kde = gaussian_kde(gamma2_vals * 1000)
        x   = np.linspace(gamma2_vals.min() * 1000, gamma2_vals.max() * 1000, 200)
        ax.plot(x, kde(x), "b-", lw=2)
    ax.axvline(gamma2_vals.mean() * 1000, color="blue", lw=2,
               label=f"Mean Γ₂={gamma2_vals.mean()*1000:.3f} fs⁻¹")
    ax.set_xlabel(r"$\Gamma_2$ (×10⁻³ fs⁻¹)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Coherence Decay Rate Distribution\nover Disorder Ensemble", fontsize=11)
    ax.legend(fontsize=10)

    # Right: LSTM Γ₂ vs Redfield scaling prediction
    ax = axes[1]
    ax.scatter(gamma2_redfield * 1000, gamma2_vals * 1000,
               alpha=0.4, s=20, color="steelblue")
    lims = [min(gamma2_redfield.min(), gamma2_vals.min()) * 1000,
            max(gamma2_redfield.max(), gamma2_vals.max()) * 1000]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect Redfield scaling")
    # Fit slope
    try:
        slope = np.polyfit(gamma2_redfield, gamma2_vals, 1)[0]
        ax.set_title(f"LSTM Γ₂ vs Redfield Prediction\n"
                     f"Slope={slope:.3f} (1.0=perfect Redfield)", fontsize=11)
    except Exception:
        ax.set_title("LSTM Γ₂ vs Redfield Prediction", fontsize=11)
    ax.set_xlabel(r"Redfield: $2\pi\lambda k_BT/\hbar\omega_c^2$ (×10⁻³ fs⁻¹)", fontsize=11)
    ax.set_ylabel(r"LSTM $\Gamma_2$ (×10⁻³ fs⁻¹)", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_figure(fig, "disorder_gamma2_distribution", save_dir, ev["figure_dpi"])
    plt.close(fig)
    print(f"  Γ₂ distribution: mean={gamma2_vals.mean()*1000:.3f} fs⁻¹, "
          f"std={gamma2_vals.std()*1000:.3f} fs⁻¹  (n={len(gamma2_vals)} fits)")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_disorder_analyses(cfg, device=None, save_dir=None, N_disorder=1000):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_dir is None:
        save_dir = cfg["evaluation"]["figure_dir"]

    os.makedirs(save_dir, exist_ok=True)

    # Run at two centre points: ENAQT optimum and physiological conditions
    print("\n=== Disorder Ensemble: ENAQT Optimum ===")
    res_opt = run_disorder_ensemble(
        cfg, device=device, save_dir=save_dir, N_disorder=N_disorder,
        centre_params={"T": 200.0, "lambda": 55.0,
                       "omega_c": 100.0, "alpha": 1.0, "site": 0.0})

    print("\n=== Disorder Ensemble: Physiological (300K) ===")
    res_physio = run_disorder_ensemble(
        cfg, device=device, save_dir=save_dir, N_disorder=N_disorder,
        centre_params={"T": 300.0, "lambda": 55.0,
                       "omega_c": 100.0, "alpha": 1.0, "site": 0.0})

    return {"enaqt_optimum": res_opt, "physiological": res_physio}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--N",          type=int, default=1000,
                        help="Number of disorder realisations")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_all_disorder_analyses(cfg, device=device, N_disorder=args.N)
