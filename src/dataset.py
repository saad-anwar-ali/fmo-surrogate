"""
dataset.py  —  PyTorch Dataset classes for the 7-site FMO surrogate.

Three dataset variants:
  FMOSequenceDataset  →  for the LSTM (full trajectory per sample)
  FMOPointDataset     →  for the PINN  ((params, t) → rho point-wise)
  FMOInverseDataset   →  for the inverse model (trajectory → params)

Parameter normalisation
-----------------------
  5 physical parameters: [T_K, lambda_cm, omega_c_cm, alpha_scale, init_site]
  Z-scored using statistics from training split only (no leakage).

Split discipline
----------------
  Entire trajectories assigned to one split — never split across time steps.

FIX (v2): FMOSequenceDataset and FMOInverseDataset now load all data into RAM
  at __init__ time instead of opening the HDF5 file on every __getitem__ call.
  The old approach caused thousands of file open/close cycles per epoch,
  leaving the GPU at 0% utilisation. Now matches FMOPointDataset behaviour.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, N_RHO


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class ParamNormaliser:
    """Z-score normaliser for the 5 physical parameters."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std  = std

    def fit(self, params: np.ndarray) -> "ParamNormaliser":
        self.mean = params.mean(axis=0).astype(np.float32)
        self.std  = (params.std(axis=0) + 1e-8).astype(np.float32)
        return self

    def transform(self, params: np.ndarray) -> np.ndarray:
        return ((params.astype(np.float32) - self.mean) / self.std)

    def inverse_transform(self, params_norm: np.ndarray) -> np.ndarray:
        return params_norm * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.array(d["mean"], dtype=np.float32),
                   std =np.array(d["std"],  dtype=np.float32))


# ---------------------------------------------------------------------------
# Sequence dataset (LSTM)  — fixed: full in-memory load
# ---------------------------------------------------------------------------

class FMOSequenceDataset(Dataset):
    """
    One sample = one complete trajectory.
    Returns (params_norm, rho_sequence, t_norm).
    All data loaded into RAM at init — no per-sample HDF5 I/O.
    """
    def __init__(self, hdf5_path, indices, normaliser):
        indices = np.asarray(indices, dtype=np.int64)
        with h5py.File(hdf5_path, "r") as f:
            t_fs       = f["trajectories/t_fs"][:]
            all_params = f["trajectories/params"][:]
            all_rho    = f["trajectories/rho_flat"][:]
        self.n_steps     = len(t_fs)
        self.t_norm      = ((t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())).astype(np.float32)
        self.params_norm = np.stack([normaliser.transform(all_params[i].astype(np.float32)) for i in indices])
        self.rho         = all_rho[indices].astype(np.float32)

    def __len__(self):
        return len(self.params_norm)

    def __getitem__(self, idx):
        return (torch.tensor(self.params_norm[idx], dtype=torch.float32),
                torch.tensor(self.rho[idx],         dtype=torch.float32),
                torch.tensor(self.t_norm,           dtype=torch.float32))


# ---------------------------------------------------------------------------
# Point dataset (PINN)
# ---------------------------------------------------------------------------

class FMOPointDataset(Dataset):
    """
    One sample = one (params, t) → rho(t) mapping.
    Total size = n_traj × n_steps.
    All data loaded into RAM at init.
    """
    def __init__(self, hdf5_path, indices, normaliser):
        self.indices = np.asarray(indices, dtype=np.int64)
        with h5py.File(hdf5_path, "r") as f:
            t_fs       = f["trajectories/t_fs"][:]
            all_params = f["trajectories/params"][:]
            all_rho    = f["trajectories/rho_flat"][:]
        self.n_steps = len(t_fs)
        self.t_norm  = ((t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())).astype(np.float32)
        self.params_norm = np.stack([
            normaliser.transform(all_params[i].astype(np.float32))
            for i in self.indices
        ])
        self.rho = all_rho[self.indices]

    def __len__(self):
        return len(self.indices) * self.n_steps

    def __getitem__(self, flat_idx):
        traj_local = flat_idx // self.n_steps
        step_idx   = flat_idx  % self.n_steps
        x = np.concatenate([self.params_norm[traj_local],
                             [self.t_norm[step_idx]]]).astype(np.float32)
        return (torch.tensor(x,                                   dtype=torch.float32),
                torch.tensor(self.rho[traj_local, step_idx],     dtype=torch.float32))


# ---------------------------------------------------------------------------
# Inverse dataset  — fixed: full in-memory load
# ---------------------------------------------------------------------------

class FMOInverseDataset(Dataset):
    """
    One sample = (rho_sequence_flat, params_normalised).
    All data loaded into RAM at init — no per-sample HDF5 I/O.
    """
    def __init__(self, hdf5_path, indices, normaliser):
        indices = np.asarray(indices, dtype=np.int64)
        with h5py.File(hdf5_path, "r") as f:
            all_params = f["trajectories/params"][:]
            all_rho    = f["trajectories/rho_flat"][:]
        self.params_norm = np.stack([normaliser.transform(all_params[i].astype(np.float32)) for i in indices])
        self.traj_flat   = all_rho[indices].astype(np.float32).reshape(len(indices), -1)

    def __len__(self):
        return len(self.params_norm)

    def __getitem__(self, idx):
        return (torch.tensor(self.traj_flat[idx],   dtype=torch.float32),
                torch.tensor(self.params_norm[idx], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_splits(cfg, hdf5_path=None):
    """Split trajectory indices into train/val/test (trajectory-level)."""
    path = hdf5_path or cfg["data"]["lindblad_hdf5"]
    with h5py.File(path, "r") as f:
        n_traj     = f["trajectories/params"].shape[0]
        all_params = f["trajectories/params"][:]
    all_idx = np.arange(n_traj)
    sp = cfg["split"]
    vt_frac = sp["val_frac"] + sp["test_frac"]
    train_idx, vt_idx = train_test_split(all_idx, test_size=vt_frac,
                                          random_state=cfg["seed"], shuffle=True)
    test_frac_of_vt = sp["test_frac"] / vt_frac
    val_idx, test_idx = train_test_split(vt_idx, test_size=test_frac_of_vt,
                                          random_state=cfg["seed"], shuffle=True)
    norm = ParamNormaliser().fit(all_params[train_idx])
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    return train_idx, val_idx, test_idx, norm


def build_dataloaders(cfg, model_type="lstm", hdf5_path=None,
                      num_workers=0, pin_memory=False, persistent_workers=False):
    """Build DataLoaders for all splits."""
    path = hdf5_path or cfg["data"]["lindblad_hdf5"]
    train_idx, val_idx, test_idx, norm = build_splits(cfg, path)
    DS = {"lstm": FMOSequenceDataset,
          "pinn": FMOPointDataset,
          "inverse": FMOInverseDataset}[model_type]
    bs = cfg["training"]["batch_size"]
    kw = dict(num_workers=num_workers, pin_memory=pin_memory,
              persistent_workers=persistent_workers)
    return (DataLoader(DS(path, train_idx, norm), batch_size=bs, shuffle=True,  **kw),
            DataLoader(DS(path, val_idx,   norm), batch_size=bs, shuffle=False, **kw),
            DataLoader(DS(path, test_idx,  norm), batch_size=bs, shuffle=False, **kw),
            norm, train_idx, val_idx, test_idx)
