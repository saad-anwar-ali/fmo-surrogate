# %% [markdown]
# # Notebook 04 — Inverse Problem: Recovering Physical Parameters
#
# This notebook analyses the normalising flow inverse model.
# Key questions:
# - How accurately can we recover [T, λ, ω_c, α] from a trajectory?
# - How does noise level affect parameter recovery?
# - What does the posterior P(params | trajectory) look like?
# - Can we detect when parameters are extrapolated beyond training range?
#
# **Run after:** `python src/train.py --model inverse`

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, h5py, torch, matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from src.utils import load_config, set_plot_style, N_RHO
from src.dataset import ParamNormaliser, build_splits
from src.evaluate import load_model, predict_inverse_posterior

set_plot_style()
cfg    = load_config("config.yaml")
device = torch.device("cpu")

inv_model, inv_norm, test_idx = load_model("inverse", cfg, device)
hdf5 = cfg["data"]["lindblad_hdf5"]
with h5py.File(hdf5, "r") as f:
    t_fs = f["trajectories/t_fs"][:]
_, _, _, norm = build_splits(cfg)
param_names_phys = ["T (K)", "λ (cm⁻¹)", "ω_c (cm⁻¹)", "α", "Init site"]
print(f"Test trajectories: {len(test_idx)}")

# %% [markdown]
# ## 1. MAP accuracy — point estimate vs true parameters

# %%
true_params, map_params = [], []
with h5py.File(hdf5, "r") as f:
    for ti in test_idx[:40]:
        gt  = f["trajectories/rho_flat"][int(ti)]
        par = f["trajectories/params"][int(ti)]
        samples_norm  = predict_inverse_posterior(inv_model, gt, n_samples=500, device=device)
        mean_norm     = samples_norm.mean(axis=0)       # posterior mean (MMSE estimate)
        mean_phys     = inv_norm.inverse_transform(mean_norm)
        true_params.append(par.astype(np.float32))
        map_params.append(mean_phys)

true_params = np.array(true_params)   # (N, 5)
map_params  = np.array(map_params)    # (N, 5)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for col, (ax, pname) in enumerate(zip(axes, param_names_phys)):
    ax.scatter(true_params[:, col], map_params[:, col], alpha=0.7, s=40, c="steelblue")
    lims = [min(true_params[:,col].min(), map_params[:,col].min())*0.95,
            max(true_params[:,col].max(), map_params[:,col].max())*1.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
    if true_params[:, col].std() > 0:
        r2 = r2_score(true_params[:, col], map_params[:, col])
        ax.set_title(f"{pname}\n$R^2={r2:.4f}$", fontsize=10)
    else:
        ax.set_title(pname, fontsize=10)
    ax.set_xlabel("True"); ax.set_ylabel("MAP estimate" if col==0 else "")
    ax.legend(fontsize=8)
plt.suptitle("Inverse Model Posterior Mean vs True Parameters\n(MMSE estimate — minimises expected squared error)", fontsize=12)
plt.tight_layout()
plt.savefig("results/figures/04_map_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Noise robustness — parameter recovery under noisy observations

# %%
noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
n_noise_test = min(10, len(test_idx))

T_errors = {nl: [] for nl in noise_levels}
lam_errors = {nl: [] for nl in noise_levels}

with h5py.File(hdf5, "r") as f:
    for ti in test_idx[:n_noise_test]:
        gt  = f["trajectories/rho_flat"][int(ti)]
        par = f["trajectories/params"][int(ti)]
        T_true   = float(par[0])
        lam_true = float(par[1])
        for nl in noise_levels:
            noise    = nl * np.random.randn(*gt.shape).astype(np.float32)
            gt_noisy = gt + noise
            samps    = predict_inverse_posterior(inv_model, gt_noisy, n_samples=200, device=device)
            map_n    = inv_norm.inverse_transform(samps.mean(axis=0))
            T_errors[nl].append(abs(map_n[0] - T_true))
            lam_errors[nl].append(abs(map_n[1] - lam_true))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nl_pct = [100*nl for nl in noise_levels]
T_mae   = [np.mean(T_errors[nl])   for nl in noise_levels]
lam_mae = [np.mean(lam_errors[nl]) for nl in noise_levels]

axes[0].plot(nl_pct, T_mae,   "bo-", lw=2, ms=8)
axes[0].set_xlabel("Noise level (%)", fontsize=12)
axes[0].set_ylabel("MAE in T (K)")
axes[0].set_title("Temperature Recovery vs Noise")

axes[1].plot(nl_pct, lam_mae, "ro-", lw=2, ms=8)
axes[1].set_xlabel("Noise level (%)")
axes[1].set_ylabel(r"MAE in $\lambda$ (cm$^{-1}$)")
axes[1].set_title(r"$\lambda$ Recovery vs Noise")

plt.tight_layout()
plt.savefig("results/figures/04_noise_robustness.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Posterior width vs parameter identifiability

# %%
# Parameters with narrower posteriors are more identifiable from the trajectory
with h5py.File(hdf5, "r") as f:
    ti  = int(test_idx[0])
    gt  = f["trajectories/rho_flat"][ti]
    par = f["trajectories/params"][ti]

samps      = predict_inverse_posterior(inv_model, gt, n_samples=2000, device=device)
samps_phys = inv_norm.inverse_transform(samps)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for col, (ax, pname) in enumerate(zip(axes, param_names_phys)):
    ax.hist(samps_phys[:, col], bins=50, density=True,
            color="steelblue", alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(par[col], color="red", lw=2.5, label=f"True={par[col]:.1f}")
    ax.set_xlabel(pname); ax.set_title(f"Posterior width={samps_phys[:,col].std():.2f}")
    ax.legend(fontsize=9)
fig.suptitle("Posterior Distributions P(param | trajectory)\n"
             "Narrower = more identifiable from quantum dynamics", fontsize=11)
plt.tight_layout()
plt.savefig("results/figures/04_posterior_widths.png", dpi=150, bbox_inches="tight")
plt.show()
print("Notebook 04 complete.")
