# %% [markdown]
# # Notebook 03 — ENAQT Landscape Reconstruction
#
# ## The key scientific result
#
# This notebook demonstrates the primary scientific contribution of the surrogate
# model: using it to **densely reconstruct the ENAQT efficiency landscape** in
# parameter space, interpolating between the sparse QuTiP simulation points.
#
# ### What is ENAQT?
#
# Environment-Assisted Quantum Transport (ENAQT) is the counterintuitive
# phenomenon where adding environmental noise (dephasing) to a quantum system
# can *improve* rather than *destroy* excitation transport.
#
# The mechanism (Rebentrost, Mohseni, Kassal, Lloyd, Aspuru-Guzik 2009):
# 1. **Zero dephasing**: the excitation is delocalised into eigenstates of H.
#    If the energy eigenstates don't couple well to the reaction centre, the
#    transfer is slow (quantum localisation / destructive interference).
# 2. **Optimal dephasing**: decoherence breaks the destructive interference,
#    allowing the excitation to explore different pathways (quantum walk → classical
#    random walk crossover), often landing in a more efficient transport regime.
# 3. **Excessive dephasing**: the quantum Zeno effect freezes the dynamics.
#    Too many measurements (interactions with the bath) slow the evolution.
#
# In the FMO complex, this optimal dephasing is believed to correspond to
# physiological conditions (T ≈ 300 K), providing a possible explanation for
# the efficiency of natural photosynthesis.
#
# **Run after** `python src/train.py --model pinn` has completed.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "src"))

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from src.utils import (
    load_config, set_plot_style, seed_everything,
    dephasing_rate_cm, compute_trace, K_B_CM_PER_K, HBAR_CM_FS,
)
from src.evaluate import load_model, predict_pinn

set_plot_style()
cfg = load_config("config.yaml")
seed_everything(cfg["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

pinn_model, pinn_norm, test_idx = load_model("pinn", cfg, device)

hdf5_path = cfg["data"]["hdf5_path"]
sweep = cfg["sweep"]

with h5py.File(hdf5_path, "r") as f:
    t_fs = f["trajectories/t_fs"][:]

t_norm  = (t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())
t_final = t_norm[-1]

print(f"Time axis: {t_fs[0]:.0f} – {t_fs[-1]:.0f} fs ({len(t_fs)} steps)")

# %% [markdown]
# ## 1. ENAQT landscape in (T, λ) space
#
# We hold ω_c and α fixed and sweep T and λ densely with the PINN.

# %%
# Fixed parameters
omega_c_fixed_cm = 100.0
alpha_fixed      = 1.0
grid_n           = cfg["evaluation"]["enaqt_grid_size"]

# Dense grid
T_vals   = np.linspace(min(sweep["T_K"]), max(sweep["T_K"]), grid_n)
lam_vals = np.linspace(min(sweep["lambda_cm"]), max(sweep["lambda_cm"]), grid_n)
T_mesh, L_mesh = np.meshgrid(T_vals, lam_vals, indexing="ij")

N = grid_n * grid_n
T_flat   = T_mesh.ravel()
L_flat   = L_mesh.ravel()
omc_flat = np.full(N, omega_c_fixed_cm, dtype=np.float32)
al_flat  = np.full(N, alpha_fixed,       dtype=np.float32)
t_flat   = np.ones(N, dtype=np.float32)

params_raw  = np.stack([T_flat, L_flat, omc_flat, al_flat], axis=-1).astype(np.float32)
params_norm = pinn_norm.transform(params_raw)
x = np.concatenate([params_norm, t_flat.reshape(-1, 1)], axis=-1)

print(f"Querying PINN on {N:,} grid points ...")
x_t = torch.tensor(x, dtype=torch.float32, device=device)
with torch.no_grad():
    rho_pred = pinn_model(x_t).cpu().numpy()

eff_grid = np.clip(1.0 - (rho_pred[:, 0] + rho_pred[:, 4] + rho_pred[:, 8]), 0, 1)
eff_grid = eff_grid.reshape(grid_n, grid_n)

# Smooth slightly for publication quality
eff_smooth = gaussian_filter(eff_grid, sigma=0.8)

# Load ground-truth points
with h5py.File(hdf5_path, "r") as f:
    all_params = f["trajectories/params"][:]
    all_eff    = f["trajectories/efficiency"][:]

mask_gt = (
    np.isclose(all_params[:, 2], omega_c_fixed_cm, atol=1.0) &
    np.isclose(all_params[:, 3], alpha_fixed,       atol=0.01)
)
gt_T   = all_params[mask_gt, 0]
gt_lam = all_params[mask_gt, 1]
gt_eff = all_eff[mask_gt]
print(f"Ground-truth overlay: {mask_gt.sum()} points")

# %% [markdown]
# ### The ENAQT heatmap

# %%
fig = plt.figure(figsize=(11, 7.5))
ax = fig.add_subplot(111)

im = ax.pcolormesh(
    T_mesh, L_mesh, eff_smooth,
    cmap="viridis", shading="auto",
    vmin=0, vmax=max(eff_smooth.max(), gt_eff.max()),
)
cbar = plt.colorbar(im, ax=ax, label="Transfer Efficiency", fraction=0.046)
cbar.ax.tick_params(labelsize=11)

# Contour lines for readability
contours = ax.contour(
    T_mesh, L_mesh, eff_smooth,
    levels=8, colors="white", alpha=0.3, linewidths=0.8
)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

# Ground-truth overlay
sc = ax.scatter(
    gt_T, gt_lam,
    c=gt_eff, cmap="viridis",
    edgecolors="white", linewidths=2.0,
    s=130, marker="D",
    vmin=0, vmax=max(eff_smooth.max(), gt_eff.max()),
    zorder=5, label="QuTiP simulations",
)
ax.legend(loc="upper right", fontsize=11, framealpha=0.9,
          facecolor="white", edgecolor="gray")

ax.set_xlabel("Temperature T (K)", fontsize=14)
ax.set_ylabel(r"Reorganisation energy $\lambda$ (cm$^{-1}$)", fontsize=14)
ax.set_title(
    "ENAQT Efficiency Landscape\n"
    r"PINN surrogate — $\omega_c = 100$ cm$^{-1}$, $\alpha = 1.0$",
    fontsize=13,
)

# Mark the optimal efficiency point
i_opt, j_opt = np.unravel_index(np.argmax(eff_smooth), eff_smooth.shape)
T_opt   = T_vals[i_opt]
lam_opt = lam_vals[j_opt]
eff_opt = eff_smooth[i_opt, j_opt]
ax.plot(T_opt, lam_opt, "r*", markersize=18, zorder=6,
        label=f"Optimal: T={T_opt:.0f}K, λ={lam_opt:.0f}")
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

print(f"\nOptimal efficiency: {eff_opt:.4f} at T={T_opt:.0f}K, λ={lam_opt:.0f} cm⁻¹")

plt.tight_layout()
plt.savefig("results/figures/03_enaqt_landscape.png", dpi=150, bbox_inches="tight")
plt.savefig("results/figures/03_enaqt_landscape.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. ENAQT landscape in (T, α) space
#
# α controls the overall dephasing coupling strength.
# We vary α and T while holding λ and ω_c fixed.

# %%
alpha_vals = np.linspace(min(sweep["alpha_scale"]), max(sweep["alpha_scale"]), grid_n)
lambda_fixed_cm = 55.0

T_mesh2, A_mesh2 = np.meshgrid(T_vals, alpha_vals, indexing="ij")
N2 = T_mesh2.size
T_flat2   = T_mesh2.ravel().astype(np.float32)
lam_flat2 = np.full(N2, lambda_fixed_cm, dtype=np.float32)
omc_flat2 = np.full(N2, omega_c_fixed_cm, dtype=np.float32)
al_flat2  = A_mesh2.ravel().astype(np.float32)
t_flat2   = np.ones(N2, dtype=np.float32)

params_raw2  = np.stack([T_flat2, lam_flat2, omc_flat2, al_flat2], axis=-1)
params_norm2 = pinn_norm.transform(params_raw2)
x2 = np.concatenate([params_norm2, t_flat2.reshape(-1, 1)], axis=-1).astype(np.float32)

x2_t = torch.tensor(x2, dtype=torch.float32, device=device)
with torch.no_grad():
    rho2 = pinn_model(x2_t).cpu().numpy()

eff_grid2 = np.clip(1.0 - (rho2[:, 0] + rho2[:, 4] + rho2[:, 8]), 0, 1)
eff_grid2 = eff_grid2.reshape(grid_n, grid_n)
eff_smooth2 = gaussian_filter(eff_grid2, sigma=0.8)

# Ground-truth overlay
mask_gt2 = (
    np.isclose(all_params[:, 1], lambda_fixed_cm, atol=1.0) &
    np.isclose(all_params[:, 2], omega_c_fixed_cm, atol=1.0)
)
gt_T2   = all_params[mask_gt2, 0]
gt_al2  = all_params[mask_gt2, 3]
gt_eff2 = all_eff[mask_gt2]

fig, ax = plt.subplots(figsize=(9, 6))
im2 = ax.pcolormesh(T_mesh2, A_mesh2, eff_smooth2, cmap="viridis", shading="auto")
plt.colorbar(im2, ax=ax, label="Transfer Efficiency")
contours2 = ax.contour(T_mesh2, A_mesh2, eff_smooth2, levels=6, colors="white", alpha=0.35, lw=0.8)
ax.clabel(contours2, inline=True, fontsize=8, fmt="%.2f")
ax.scatter(gt_T2, gt_al2, c=gt_eff2, cmap="viridis",
           edgecolors="white", linewidths=1.5, s=100, marker="D", zorder=5, label="QuTiP")
ax.legend(fontsize=10)
ax.set_xlabel("Temperature T (K)", fontsize=13)
ax.set_ylabel(r"Dephasing scale $\alpha$", fontsize=13)
ax.set_title(
    r"ENAQT Landscape: T vs $\alpha$" + "\n"
    fr"(λ={lambda_fixed_cm:.0f} cm⁻¹, ω_c={omega_c_fixed_cm:.0f} cm⁻¹)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("results/figures/03_enaqt_T_alpha.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. 1D slices: efficiency vs dephasing rate (ENAQT curve)
#
# The classic ENAQT curve shows efficiency as a function of dephasing rate γ_φ.
# We generate this by fixing T and sweeping λ (since γ_φ ∝ λ·T).

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: efficiency vs γ_φ for each T
T_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sweep["T_K"])))

for T_val, col in zip(sweep["T_K"], T_colors):
    # Sweep λ densely at this T
    lam_sweep = np.linspace(min(sweep["lambda_cm"]), max(sweep["lambda_cm"]), 80)
    omc_s  = np.full(80, omega_c_fixed_cm, dtype=np.float32)
    al_s   = np.full(80, alpha_fixed,       dtype=np.float32)
    T_s    = np.full(80, T_val,             dtype=np.float32)
    t_s    = np.ones(80, dtype=np.float32)

    params_s = np.stack([T_s, lam_sweep.astype(np.float32), omc_s, al_s], axis=-1)
    params_sn = pinn_norm.transform(params_s)
    xs = np.concatenate([params_sn, t_s.reshape(-1, 1)], axis=-1).astype(np.float32)

    with torch.no_grad():
        rho_s = pinn_model(torch.tensor(xs, device=device)).cpu().numpy()
    eff_s = np.clip(1.0 - (rho_s[:, 0] + rho_s[:, 4] + rho_s[:, 8]), 0, 1)

    # Compute dephasing rates
    gp_s = np.array([dephasing_rate_cm(T_val, l, omega_c_fixed_cm, alpha_fixed)
                     for l in lam_sweep])

    axes[0].plot(gp_s, eff_s, "-", color=col, lw=2.0, label=f"T={T_val:.0f}K")

    # Overlay QuTiP points
    mask_t = (
        (all_params[:, 0] == T_val) &
        np.isclose(all_params[:, 2], omega_c_fixed_cm, atol=1.0) &
        np.isclose(all_params[:, 3], alpha_fixed, atol=0.01)
    )
    gp_gt = np.array([dephasing_rate_cm(T_val, l, omega_c_fixed_cm, alpha_fixed)
                      for l in all_params[mask_t, 1]])
    axes[0].scatter(gp_gt, all_eff[mask_t], color=col, s=60, zorder=5, edgecolors="black", lw=0.8)

axes[0].set_xlabel(r"Dephasing rate $\gamma_\phi$ (cm$^{-1}$)", fontsize=12)
axes[0].set_ylabel("Transfer Efficiency", fontsize=12)
axes[0].set_title("ENAQT Curve: Efficiency vs Dephasing Rate\n(Solid=PINN, Dots=QuTiP)", fontsize=11)
axes[0].legend(fontsize=9)

# Right: efficiency vs T for each λ
lambda_vals_plot = [10, 40, 70, 100]
lam_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambda_vals_plot)))

for lam_val, col in zip(lambda_vals_plot, lam_colors):
    T_sweep = np.linspace(min(sweep["T_K"]), max(sweep["T_K"]), 80)
    lam_sw  = np.full(80, lam_val,           dtype=np.float32)
    omc_sw  = np.full(80, omega_c_fixed_cm,  dtype=np.float32)
    al_sw   = np.full(80, alpha_fixed,        dtype=np.float32)
    t_sw    = np.ones(80, dtype=np.float32)

    params_sw  = np.stack([T_sweep.astype(np.float32), lam_sw, omc_sw, al_sw], axis=-1)
    params_swn = pinn_norm.transform(params_sw)
    xsw = np.concatenate([params_swn, t_sw.reshape(-1, 1)], axis=-1).astype(np.float32)

    with torch.no_grad():
        rho_sw = pinn_model(torch.tensor(xsw, device=device)).cpu().numpy()
    eff_sw = np.clip(1.0 - (rho_sw[:, 0] + rho_sw[:, 4] + rho_sw[:, 8]), 0, 1)

    axes[1].plot(T_sweep, eff_sw, "-", color=col, lw=2.0, label=f"λ={lam_val:.0f} cm⁻¹")

axes[1].set_xlabel("Temperature T (K)", fontsize=12)
axes[1].set_ylabel("Transfer Efficiency", fontsize=12)
axes[1].set_title("Efficiency vs Temperature (fixed λ values)\nPINN surrogate", fontsize=11)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/figures/03_enaqt_1d_slices.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Surrogate accuracy in the ENAQT landscape
#
# We directly compare PINN efficiency predictions vs QuTiP at the grid
# points used in the original sweep.

# %%
# Compute PINN efficiency for all training data points
print("Computing PINN efficiency for all dataset trajectories ...")

all_params_np = all_params.copy()
params_norm_all = pinn_norm.transform(all_params_np.astype(np.float32))
t_final_arr     = np.ones(len(all_params_np), dtype=np.float32)
x_all = np.concatenate([params_norm_all, t_final_arr.reshape(-1, 1)], axis=-1).astype(np.float32)

BATCH = 256
pinn_eff_all = []
for i in range(0, len(x_all), BATCH):
    xb = torch.tensor(x_all[i:i+BATCH], dtype=torch.float32, device=device)
    with torch.no_grad():
        rb = pinn_model(xb).cpu().numpy()
    eff_b = np.clip(1.0 - (rb[:, 0] + rb[:, 4] + rb[:, 8]), 0, 1)
    pinn_eff_all.extend(eff_b.tolist())

pinn_eff_all = np.array(pinn_eff_all)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Parity plot
axes[0].scatter(all_eff, pinn_eff_all, alpha=0.4, s=18, c="steelblue", edgecolors="none")
lim = [min(all_eff.min(), pinn_eff_all.min())*0.95,
       max(all_eff.max(), pinn_eff_all.max())*1.05]
axes[0].plot(lim, lim, "r--", lw=1.5, label="Perfect")

from sklearn.metrics import r2_score
r2 = r2_score(all_eff, pinn_eff_all)
axes[0].set_xlabel("QuTiP Transfer Efficiency")
axes[0].set_ylabel("PINN Predicted Efficiency")
axes[0].set_title(f"Efficiency Prediction Accuracy\n$R^2 = {r2:.4f}$")
axes[0].legend()

# Error distribution
err = pinn_eff_all - all_eff
axes[1].hist(err, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", lw=1.5, ls="--")
axes[1].axvline(err.mean(), color="orange", lw=1.5, label=f"Mean={err.mean():.4f}")
axes[1].set_xlabel("PINN − QuTiP efficiency")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Prediction Error Distribution\nSTD={err.std():.4f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/figures/03_efficiency_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nEfficiency prediction summary:")
print(f"  R² = {r2:.4f}")
print(f"  MAE = {np.abs(err).mean():.5f}")
print(f"  RMSE = {np.sqrt((err**2).mean()):.5f}")
print(f"\nNotebook 03 complete.")
