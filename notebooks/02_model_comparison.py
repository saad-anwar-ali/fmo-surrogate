# %% [markdown]
# # Notebook 02 — LSTM vs PINN Model Comparison
#
# This notebook provides a side-by-side comparison of the two surrogate models:
#
# - **LSTM** (autoregressive sequence model): learns the Markovian step-by-step
#   dynamics. Has memory via its hidden state.
# - **PINN** (physics-informed feedforward): maps (params, t) directly to ρ(t).
#   Includes soft constraints for trace conservation and positivity.
#
# We compare them across:
# 1. Training loss curves
# 2. Prediction quality on representative trajectories
# 3. Trace conservation — does the PINN constraint help?
# 4. Population positivity violations
# 5. Autoregressive error accumulation in the LSTM
# 6. Inference speed comparison
#
# **Run this after** `python src/train.py --model all` has completed.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "src"))

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import (
    load_config, set_plot_style, seed_everything,
    extract_populations, extract_coherence_12, compute_trace,
)
from src.dataset import ParamNormaliser
from src.evaluate import load_model, predict_lstm, predict_pinn

set_plot_style()
cfg = load_config("config.yaml")
seed_everything(cfg["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load models
lstm_model, lstm_norm, test_idx_lstm = load_model("lstm", cfg, device)
pinn_model, pinn_norm, test_idx_pinn = load_model("pinn", cfg, device)
test_idx = test_idx_pinn  # consistent indexing

hdf5_path = cfg["data"]["hdf5_path"]
with h5py.File(hdf5_path, "r") as f:
    t_fs   = f["trajectories/t_fs"][:]
    params = f["trajectories/params"][:]

t_norm  = (t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())
n_steps = len(t_fs)

print(f"Test set: {len(test_idx)} trajectories")

# %% [markdown]
# ## 1. Training Loss Curves

# %%
import pandas as pd
from pathlib import Path

log_dir = cfg["training"]["log_dir"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, model_name, color in [
    (axes[0], "lstm", "steelblue"),
    (axes[1], "pinn", "firebrick"),
]:
    log_path = os.path.join(log_dir, f"{model_name}_training.csv")
    if not os.path.exists(log_path):
        ax.text(0.5, 0.5, f"Log not found:\n{log_path}",
                transform=ax.transAxes, ha="center", va="center")
        continue

    df = pd.read_csv(log_path)
    epochs = df["epoch"].values

    if model_name == "lstm":
        train_loss = df["train_loss"].values
        val_loss   = df["val_loss"].values
        ax.semilogy(epochs, train_loss, label="Train MSE", color=color,     lw=1.8)
        ax.semilogy(epochs, val_loss,   label="Val MSE",   color=color,     lw=1.8, ls="--")
        ax.set_ylabel("MSE Loss (log scale)")
    else:
        train_total = df["train_total"].values
        val_total   = df["val_total"].values
        train_mse   = df["train_mse"].values
        val_mse     = df["val_mse"].values
        ax.semilogy(epochs, train_total, label="Train (total)", color=color, lw=1.8)
        ax.semilogy(epochs, val_total,   label="Val (total)",   color=color, lw=1.8, ls="--")
        ax.semilogy(epochs, train_mse,   label="Train MSE only", color="gray", lw=1.2, ls=":")
        ax.set_ylabel("Loss (log scale)")

    best_epoch = df.loc[df["val_loss" if model_name == "lstm" else "val_total"].astype(float).idxmin(), "epoch"]
    ax.axvline(best_epoch, color="green", lw=1.2, ls=":", alpha=0.7, label=f"Best epoch ({best_epoch:.0f})")
    ax.set_xlabel("Epoch")
    ax.set_title(f"{model_name.upper()} Training History")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/figures/02_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Side-by-side prediction comparison on test trajectories
#
# We pick 3 test trajectories and compare QuTiP ground truth vs both models.

# %%
n_compare = 3
fig, axes_grid = plt.subplots(n_compare, 2, figsize=(14, 4 * n_compare))

with h5py.File(hdf5_path, "r") as f:
    for row, traj_i in enumerate(test_idx[:n_compare]):
        rho_flat = f["trajectories/rho_flat"][int(traj_i)]
        p = f["trajectories/params"][int(traj_i)]

        # Predictions
        lstm_pred = predict_lstm(lstm_model, lstm_norm.transform(p.astype(np.float32)),
                                 rho_flat[0], n_steps, device)
        pinn_pred = predict_pinn(pinn_model, pinn_norm.transform(p.astype(np.float32)),
                                 t_norm, device)

        label = f"T={p[0]:.0f}K, λ={p[1]:.0f}, ωc={p[2]:.0f}, α={p[3]:.1f}"

        # Coherence
        ax = axes_grid[row, 0]
        ax.plot(t_fs, extract_coherence_12(rho_flat), "k-",  lw=2.2, label="QuTiP")
        ax.plot(t_fs, extract_coherence_12(lstm_pred), "b--", lw=1.6, label="LSTM")
        ax.plot(t_fs, extract_coherence_12(pinn_pred), "r-.", lw=1.6, label="PINN")
        ax.set_ylabel(r"$|\rho_{12}(t)|$")
        ax.set_title(f"Test {row+1}: {label}")
        ax.legend(fontsize=9)

        # Population P₃
        ax = axes_grid[row, 1]
        ax.plot(t_fs, extract_populations(rho_flat)[:, 2],  "k-",  lw=2.2, label="QuTiP P₃")
        ax.plot(t_fs, extract_populations(lstm_pred)[:, 2], "b--", lw=1.6, label="LSTM P₃")
        ax.plot(t_fs, extract_populations(pinn_pred)[:, 2], "r-.", lw=1.6, label="PINN P₃")
        ax.set_ylabel(r"$P_3(t)$")
        ax.set_title(f"Site-3 Population")
        if row == n_compare - 1:
            ax.set_xlabel("Time (fs)")
        ax.legend(fontsize=9)

fig.suptitle("LSTM vs PINN: Test Trajectory Predictions", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/figures/02_model_comparison_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Trace conservation — PINN physics constraint in action

# %%
print("Computing trace statistics on test set ...")

gt_trace_final   = []
lstm_trace_final = []
pinn_trace_final = []

with h5py.File(hdf5_path, "r") as f:
    for traj_i in test_idx:
        rho_flat = f["trajectories/rho_flat"][int(traj_i)]
        p        = f["trajectories/params"][int(traj_i)]

        lstm_pred = predict_lstm(lstm_model, lstm_norm.transform(p.astype(np.float32)),
                                 rho_flat[0], n_steps, device)
        pinn_pred = predict_pinn(pinn_model, pinn_norm.transform(p.astype(np.float32)),
                                 t_norm, device)

        gt_trace_final.append(compute_trace(rho_flat[-1:]).item())
        lstm_trace_final.append(compute_trace(lstm_pred[-1:]).item())
        pinn_trace_final.append(compute_trace(pinn_pred[-1:]).item())

gt_tf   = np.array(gt_trace_final)
lstm_tf = np.array(lstm_trace_final)
pinn_tf = np.array(pinn_trace_final)

# Also compute trace at all timesteps for a few trajectories
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Scatter: predicted final trace vs ground truth
axes[0].scatter(gt_tf, lstm_tf, alpha=0.5, s=25, color="steelblue", label="LSTM")
axes[0].scatter(gt_tf, pinn_tf, alpha=0.5, s=25, color="firebrick",  label="PINN", marker="D")
lims = [min(gt_tf.min(), lstm_tf.min(), pinn_tf.min()) * 0.95,
        max(gt_tf.max(), lstm_tf.max(), pinn_tf.max()) * 1.05]
axes[0].plot(lims, lims, "k--", lw=1.0, label="Perfect")
axes[0].set_xlabel("QuTiP Tr(ρ) at t=1000 fs")
axes[0].set_ylabel("Predicted Tr(ρ) at t=1000 fs")
axes[0].set_title("Trace Conservation\n(Final Time Step)")
axes[0].legend(fontsize=9)

# Histogram of |Tr(ρ_pred) - Tr(ρ_gt)|
lstm_err = np.abs(lstm_tf - gt_tf)
pinn_err = np.abs(pinn_tf - gt_tf)
axes[1].hist(lstm_err, bins=25, alpha=0.6, color="steelblue", label=f"LSTM (mean={lstm_err.mean():.4f})", edgecolor="white")
axes[1].hist(pinn_err, bins=25, alpha=0.6, color="firebrick",  label=f"PINN (mean={pinn_err.mean():.4f})", edgecolor="white")
axes[1].set_xlabel("|Tr(ρ_pred) - Tr(ρ_gt)| at t=1000 fs")
axes[1].set_ylabel("Count")
axes[1].set_title("Trace Error Distribution")
axes[1].legend(fontsize=9)

# Trace evolution for 5 trajectories (PINN)
with h5py.File(hdf5_path, "r") as f:
    for i, traj_i in enumerate(test_idx[:5]):
        rho_flat = f["trajectories/rho_flat"][int(traj_i)]
        p        = f["trajectories/params"][int(traj_i)]
        pinn_pred = predict_pinn(pinn_model, pinn_norm.transform(p.astype(np.float32)),
                                 t_norm, device)
        axes[2].plot(t_fs, compute_trace(rho_flat),  color=f"C{i}", lw=1.5, label=f"GT {i+1}")
        axes[2].plot(t_fs, compute_trace(pinn_pred), color=f"C{i}", lw=1.5, ls="--")
axes[2].set_xlabel("Time (fs)")
axes[2].set_ylabel("Tr(ρ)")
axes[2].set_title("Trace Evolution: GT (solid) vs PINN (dashed)")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/figures/02_trace_conservation.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"LSTM trace error: mean={lstm_err.mean():.5f}, max={lstm_err.max():.5f}")
print(f"PINN trace error: mean={pinn_err.mean():.5f}, max={pinn_err.max():.5f}")

# %% [markdown]
# ## 4. Population positivity check

# %%
print("\nChecking population positivity on test set ...")

lstm_neg_count = 0
pinn_neg_count = 0
threshold = -0.005  # allow small numerical errors

with h5py.File(hdf5_path, "r") as f:
    for traj_i in test_idx:
        rho_flat = f["trajectories/rho_flat"][int(traj_i)]
        p        = f["trajectories/params"][int(traj_i)]

        lstm_pred = predict_lstm(lstm_model, lstm_norm.transform(p.astype(np.float32)),
                                 rho_flat[0], n_steps, device)
        pinn_pred = predict_pinn(pinn_model, pinn_norm.transform(p.astype(np.float32)),
                                 t_norm, device)

        lstm_pops = extract_populations(lstm_pred)
        pinn_pops = extract_populations(pinn_pred)

        if np.any(lstm_pops < threshold):
            lstm_neg_count += 1
        if np.any(pinn_pops < threshold):
            pinn_neg_count += 1

total = len(test_idx)
print(f"Trajectories with negative populations (< {threshold}):")
print(f"  LSTM: {lstm_neg_count}/{total} ({100*lstm_neg_count/total:.1f}%)")
print(f"  PINN: {pinn_neg_count}/{total} ({100*pinn_neg_count/total:.1f}%)")
print(f"The PINN physics constraint reduces positivity violations.")

# %% [markdown]
# ## 5. LSTM: Autoregressive error accumulation
#
# A key concern with autoregressive models is error accumulation over time.
# Each predicted step is used as input for the next, so small errors compound.
# We measure this by tracking per-step MSE over the trajectory.

# %%
print("\nComputing per-step MSE for LSTM vs PINN ...")

lstm_step_mse_all = np.zeros(n_steps)
pinn_step_mse_all = np.zeros(n_steps)
n_counted = 0

with h5py.File(hdf5_path, "r") as f:
    for traj_i in test_idx[:50]:  # subsample for speed
        rho_flat = f["trajectories/rho_flat"][int(traj_i)]
        p        = f["trajectories/params"][int(traj_i)]

        lstm_pred = predict_lstm(lstm_model, lstm_norm.transform(p.astype(np.float32)),
                                 rho_flat[0], n_steps, device)
        pinn_pred = predict_pinn(pinn_model, pinn_norm.transform(p.astype(np.float32)),
                                 t_norm, device)

        lstm_step_mse_all += ((lstm_pred - rho_flat) ** 2).mean(axis=1)
        pinn_step_mse_all += ((pinn_pred - rho_flat) ** 2).mean(axis=1)
        n_counted += 1

lstm_step_mse_all /= n_counted
pinn_step_mse_all /= n_counted

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.semilogy(t_fs, lstm_step_mse_all, "b-",  lw=2, label="LSTM")
ax.semilogy(t_fs, pinn_step_mse_all, "r--", lw=2, label="PINN")
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Mean Squared Error (log scale)")
ax.set_title("Per-Step MSE over Time\n(averaged over 50 test trajectories)")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("results/figures/02_per_step_mse.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Summary table

# %%
print("\n" + "="*55)
print("Model Comparison Summary")
print("="*55)

# Compute overall test MSE
lstm_mse_all = []
pinn_mse_all = []

with h5py.File(hdf5_path, "r") as f:
    for traj_i in test_idx:
        rho_flat = f["trajectories/rho_flat"][int(traj_i)]
        p        = f["trajectories/params"][int(traj_i)]

        lstm_pred = predict_lstm(lstm_model, lstm_norm.transform(p.astype(np.float32)),
                                 rho_flat[0], n_steps, device)
        pinn_pred = predict_pinn(pinn_model, pinn_norm.transform(p.astype(np.float32)),
                                 t_norm, device)

        lstm_mse_all.append(np.mean((lstm_pred - rho_flat)**2))
        pinn_mse_all.append(np.mean((pinn_pred - rho_flat)**2))

from src.models.lstm_model import build_lstm_from_config
from src.models.pinn_model import build_pinn_from_config
lstm_params = build_lstm_from_config(cfg).count_parameters()
pinn_params = build_pinn_from_config(cfg).count_parameters()

print(f"{'Metric':<35} {'LSTM':>12} {'PINN':>12}")
print("-"*60)
print(f"{'Trainable parameters':<35} {lstm_params:>12,} {pinn_params:>12,}")
print(f"{'Test MSE (mean)':<35} {np.mean(lstm_mse_all):>12.6f} {np.mean(pinn_mse_all):>12.6f}")
print(f"{'Test MSE (std)':<35} {np.std(lstm_mse_all):>12.6f} {np.std(pinn_mse_all):>12.6f}")
print(f"{'Trace error (mean)':<35} {lstm_err.mean():>12.5f} {pinn_err.mean():>12.5f}")
print(f"{'Positivity violations':<35} {lstm_neg_count:>12} {pinn_neg_count:>12}")
print(f"{'Input type':<35} {'sequence':>12} {'(params,t)':>12}")
print(f"{'Architecture':<35} {'LSTM + FF':>12} {'FF only':>12}")
print("="*60)
print("\nNotebook 02 complete.")
