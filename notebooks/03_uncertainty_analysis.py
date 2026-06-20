# %% [markdown]
# # Notebook 03 — Uncertainty Quantification Analysis
#
# This notebook analyses the bootstrap ensemble uncertainty estimates.
# We examine:
# - Calibration: does the 90% CI contain ~90% of ground-truth values?
# - Sharpness: how tight are the confidence bands?
# - When does uncertainty peak? (physically: long times, mixed states)
# - Epistemic vs aleatoric decomposition
#
# **Run after:** `python src/train.py --model ensemble`

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, h5py, torch, matplotlib.pyplot as plt, pickle
from src.utils import load_config, set_plot_style, extract_coherence_12, extract_populations
from src.dataset import build_splits
from src.uncertainty import BootstrapEnsemble, compute_coverage

set_plot_style()
cfg    = load_config("config.yaml")
device = torch.device("cpu")

# Load ensemble
ens_path = "results/checkpoints/ensemble.pkl"
ens = BootstrapEnsemble(cfg, device)
ens.load(ens_path)
print(f"Ensemble size: {len(ens.models)} models")

hdf5 = cfg["data"]["lindblad_hdf5"]
with h5py.File(hdf5, "r") as f:
    t_fs = f["trajectories/t_fs"][:]

_, _, _, norm = build_splits(cfg)
import torch; from sklearn.model_selection import train_test_split
n_traj = None
with h5py.File(hdf5, "r") as f:
    n_traj = f["trajectories/params"].shape[0]
sp = cfg["split"]
all_idx = np.arange(n_traj)
vt = sp["val_frac"]+sp["test_frac"]
_, vt_idx  = train_test_split(all_idx, test_size=vt, random_state=cfg["seed"])
tf = sp["test_frac"]/vt
_, test_idx = train_test_split(vt_idx, test_size=tf, random_state=cfg["seed"])
print(f"Test trajectories: {len(test_idx)}")

# %% [markdown]
# ## 1. Calibration curves

# %%
# Compute coverage for coherence across multiple percentiles
percentile_levels = [50, 60, 70, 80, 90, 95]
empirical_coverage = []

n_eval = min(20, len(test_idx))
all_preds_coh = []
all_gt_coh    = []

with h5py.File(hdf5, "r") as f:
    for ti in test_idx[:n_eval]:
        gt  = f["trajectories/rho_flat"][int(ti)]
        par = f["trajectories/params"][int(ti)]
        pn  = norm.transform(par.astype(np.float32))
        uq  = ens.predict_with_uncertainty(pn[None], gt[None, 0], n_steps=len(t_fs))
        all_preds_coh.append(uq["all_samples"][0])   # (n_total, T, N_RHO)
        all_gt_coh.append(gt)

# Compute coherence for each sample
all_preds_coh_vals = np.array([[extract_coherence_12(s) for s in samps]
                                for samps in all_preds_coh])  # (n_eval, n_total, T)
all_gt_coh_vals    = np.array([extract_coherence_12(gt) for gt in all_gt_coh])   # (n_eval, T)

for pct in percentile_levels:
    lo = np.percentile(all_preds_coh_vals, (100-pct)/2,    axis=1)  # (n_eval, T)
    hi = np.percentile(all_preds_coh_vals, 100-(100-pct)/2, axis=1)
    inside = ((all_gt_coh_vals >= lo) & (all_gt_coh_vals <= hi)).mean()
    empirical_coverage.append(float(inside))
    print(f"  Nominal {pct}% CI → Empirical coverage: {100*inside:.1f}%")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(percentile_levels, [p for p in percentile_levels], "k--", lw=1.5, label="Perfect calibration")
ax.plot(percentile_levels, [100*c for c in empirical_coverage], "bo-", lw=2, ms=8, label="Empirical")
ax.fill_between(percentile_levels, [p-5 for p in percentile_levels],
                [p+5 for p in percentile_levels], alpha=0.1, color="gray", label="±5% tolerance")
ax.set_xlabel("Nominal coverage (%)", fontsize=12)
ax.set_ylabel("Empirical coverage (%)", fontsize=12)
ax.set_title("Calibration Curve — Bootstrap Ensemble\n(Well-calibrated = on the diagonal)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/figures/03_calibration_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Uncertainty vs time — when does it peak?

# %%
# Mean std over test trajectories vs time
all_stds = []
with h5py.File(hdf5, "r") as f:
    for ti in test_idx[:n_eval]:
        gt  = f["trajectories/rho_flat"][int(ti)]
        par = f["trajectories/params"][int(ti)]
        pn  = norm.transform(par.astype(np.float32))
        uq  = ens.predict_with_uncertainty(pn[None], gt[None, 0], n_steps=len(t_fs))
        # Std over density matrix elements
        all_stds.append(uq["std"][0].mean(axis=-1))  # (T,)

all_stds = np.array(all_stds)   # (n_eval, T)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_fs, all_stds.mean(axis=0),   "b-",  lw=2, label="Mean std (test set)")
ax.fill_between(t_fs,
                all_stds.mean(0) - all_stds.std(0),
                all_stds.mean(0) + all_stds.std(0),
                alpha=0.2, color="blue", label="±1 std across trajectories")
ax.set_xlabel("Time (fs)"); ax.set_ylabel("Prediction Std (uncertainty)")
ax.set_title("Ensemble Uncertainty vs Time\n"
             "Uncertainty grows at long times as coherences decay", fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/03_uncertainty_vs_time.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Peak uncertainty at t = {t_fs[np.argmax(all_stds.mean(0))]:.0f} fs")

# %% [markdown]
# ## 3. Epistemic vs aleatoric decomposition

# %%
# Epistemic: variance across bootstrap models (MC samples averaged per model)
# Aleatoric: mean variance within each model across MC samples

with h5py.File(hdf5, "r") as f:
    ti   = int(test_idx[0])
    gt   = f["trajectories/rho_flat"][ti]
    par  = f["trajectories/params"][ti]
pn = norm.transform(par.astype(np.float32))

n_bootstrap = len(ens.models)
n_mc        = ens.mc_samples
n_steps     = len(t_fs)

# Collect all raw predictions: (n_bootstrap, n_mc, n_steps, N_RHO)
preds_raw = np.zeros((n_bootstrap, n_mc, n_steps, 98), dtype=np.float32)
params_t  = torch.tensor(pn[None], dtype=torch.float32)
rho0_t    = torch.tensor(gt[None, 0], dtype=torch.float32)
for bi, model in enumerate(ens.models):
    model.train()
    for mc in range(n_mc):
        traj = model.rollout(params_t, rho0_t, n_steps)
        preds_raw[bi, mc] = traj.squeeze(0).detach().numpy()

# Epistemic variance: variance of model means
model_means = preds_raw.mean(axis=1)           # (n_bootstrap, T, D)
epistemic   = model_means.var(axis=0)          # (T, D)

# Aleatoric variance: mean of within-model variances
within_var  = preds_raw.var(axis=1)            # (n_bootstrap, T, D)
aleatoric   = within_var.mean(axis=0)          # (T, D)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].semilogy(t_fs, epistemic.mean(axis=-1), "b-",  lw=2, label="Epistemic (model uncertainty)")
axes[0].semilogy(t_fs, aleatoric.mean(axis=-1), "r--", lw=2, label="Aleatoric (data noise)")
axes[0].set_xlabel("Time (fs)"); axes[0].set_ylabel("Variance (log scale)")
axes[0].set_title("Epistemic vs Aleatoric Uncertainty")
axes[0].legend()

# Fraction epistemic
total = epistemic + aleatoric + 1e-30
axes[1].plot(t_fs, (epistemic / total).mean(axis=-1), "b-", lw=2)
axes[1].axhline(0.5, color="gray", ls="--", lw=1, label="50% line")
axes[1].set_xlabel("Time (fs)"); axes[1].set_ylabel("Epistemic fraction")
axes[1].set_title("Fraction of Uncertainty That Is Epistemic\n(1 = all model uncertainty, 0 = all data uncertainty)")
axes[1].set_ylim(0, 1); axes[1].legend()
plt.tight_layout()
plt.savefig("results/figures/03_epistemic_vs_aleatoric.png", dpi=150, bbox_inches="tight")
plt.show()
print("Notebook 03 complete.")
