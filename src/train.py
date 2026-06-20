"""
train.py  —  GPU-optimised training dispatcher for all four model types.

GPU changes vs CPU version:
  - DataLoader: num_workers + pin_memory from config (faster data pipeline)
  - AMP (automatic mixed precision): torch.amp for ~2x GPU throughput
  - Gradient scaler for stable AMP training
  - torch.compile() on PyTorch 2.x for additional ~20% speedup
  - TF32 enabled via torch.set_float32_matmul_precision("high")
  - Larger batch sizes handled transparently (set in config)
  - Proper CUDA memory management (empty cache between models)

Fixes applied:
  - torch.amp.GradScaler / autocast (replaces deprecated torch.cuda.amp)
  - FMOSequenceDataset + FMOInverseDataset load data into RAM (was per-sample HDF5)
  - torch.compile() state_dict loaded via _orig_mod to avoid key mismatch
  - TF32 matmul precision set at startup

Usage
-----
    python src/train.py --model lstm     --config config.yaml
    python src/train.py --model pinn     --config config.yaml
    python src/train.py --model inverse  --config config.yaml
    python src/train.py --model ensemble --config config.yaml
    python src/train.py --model all      --config config.yaml
"""

import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, seed_everything, CSVLogger, N_RHO
from dataset import build_dataloaders
from models.lstm_model    import build_lstm_from_config
from models.pinn_model    import build_pinn_from_config, compute_total_loss
from models.inverse_model import build_inverse_from_config


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  |  VRAM: {props.total_memory/1e9:.1f} GB  "
              f"|  CUDA {torch.version.cuda}")
        return dev
    print("WARNING: CUDA not available — falling back to CPU")
    return torch.device("cpu")


def maybe_compile(model, device):
    """Apply torch.compile on PyTorch 2.x + CUDA for free ~20% speedup."""
    if (device.type == "cuda" and
            hasattr(torch, "compile") and
            int(torch.__version__.split(".")[0]) >= 2):
        print("  Applying torch.compile()...")
        return torch.compile(model, mode="reduce-overhead")
    return model


def _dataloader_kwargs(cfg):
    """Extra DataLoader kwargs for GPU (num_workers + pin_memory)."""
    tc = cfg["training"]
    if torch.cuda.is_available():
        return {
            "num_workers": tc.get("num_workers", 4),
            "pin_memory":  tc.get("pin_memory", True),
            "persistent_workers": tc.get("num_workers", 4) > 0,
        }
    return {}


def _make_scheduler(opt, cfg):
    sc = cfg["training"]["lr_scheduler"]
    return ReduceLROnPlateau(opt, mode="min", factor=sc["factor"],
                             patience=sc["patience"], min_lr=sc["min_lr"])


def _save_ckpt(path, model, norm, cfg, model_type, epoch, val_loss,
               train_idx, val_idx, test_idx):
    # Save underlying module if compiled
    state = (model._orig_mod.state_dict()
             if hasattr(model, "_orig_mod") else model.state_dict())
    torch.save({
        "model_state": state,
        "normaliser":  norm.to_dict(),
        "config":      cfg,
        "model_type":  model_type,
        "epoch":       epoch,
        "val_loss":    val_loss,
        "train_idx":   train_idx.tolist(),
        "val_idx":     val_idx.tolist(),
        "test_idx":    test_idx.tolist(),
    }, path)


def _load_state(model, ckpt_d):
    """Load state dict into compiled or uncompiled model safely."""
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(ckpt_d["model_state"])


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def _lstm_epoch(model, loader, device, scaler=None, optimiser=None):
    training = optimiser is not None
    model.train() if training else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    use_amp = (scaler is not None) and (device.type == "cuda")
    with ctx:
        for params, rho_seq, _ in loader:
            params  = params.to(device, non_blocking=True)
            rho_seq = rho_seq.to(device, non_blocking=True)
            if use_amp:
                with autocast("cuda"):
                    pred = model(params, rho_seq)
                    loss = F.mse_loss(pred, rho_seq[:, 1:, :])
                if training:
                    optimiser.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
            else:
                pred = model(params, rho_seq)
                loss = F.mse_loss(pred, rho_seq[:, 1:, :])
                if training:
                    optimiser.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


def train_lstm(cfg):
    seed_everything(cfg["seed"])
    tc     = cfg["training"]
    device = get_device()
    print(f"\n{'='*60}\nTraining LSTM\n{'='*60}")

    dl_kw  = _dataloader_kwargs(cfg)
    tr_ld, va_ld, te_ld, norm, ti, vi, xi = build_dataloaders(cfg, "lstm", **dl_kw)
    model  = build_lstm_from_config(cfg).to(device)
    model  = maybe_compile(model, device)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"LSTM: {n_p:,} parameters")

    opt    = Adam(model.parameters(), lr=tc["learning_rate"],
                  weight_decay=tc["weight_decay"])
    sch    = _make_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    os.makedirs(tc["checkpoint_dir"], exist_ok=True)
    os.makedirs(tc["log_dir"],        exist_ok=True)
    ckpt = os.path.join(tc["checkpoint_dir"], "lstm_best.pt")
    log  = CSVLogger(os.path.join(tc["log_dir"], "lstm_training.csv"),
                     ["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    best, pat, t0 = float("inf"), 0, time.perf_counter()
    for ep in range(1, tc["max_epochs"] + 1):
        tr_l = _lstm_epoch(model, tr_ld, device, scaler, opt)
        va_l = _lstm_epoch(model, va_ld, device)
        sch.step(va_l)
        lr = opt.param_groups[0]["lr"]
        log.log({"epoch": ep, "train_loss": f"{tr_l:.6f}",
                 "val_loss": f"{va_l:.6f}", "lr": f"{lr:.2e}",
                 "elapsed_s": f"{time.perf_counter()-t0:.1f}"})
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:4d} | train={tr_l:.6f} val={va_l:.6f} lr={lr:.2e} "
                  f"[{time.perf_counter()-t0:.0f}s]")
        if va_l < best:
            best = va_l; pat = 0
            _save_ckpt(ckpt, model, norm, cfg, "lstm", ep, best, ti, vi, xi)
        else:
            pat += 1
            if pat >= tc["early_stopping_patience"]:
                print(f"Early stop at epoch {ep}"); break

    print(f"Best val MSE: {best:.6f}  →  {ckpt}")
    ckpt_d = torch.load(ckpt, map_location=device)
    _load_state(model, ckpt_d)
    te_l = _lstm_epoch(model, te_ld, device)
    print(f"Test MSE: {te_l:.6f}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------

def _pinn_epoch(model, loader, device, cfg, scaler=None, optimiser=None, epoch=0):
    training = optimiser is not None
    model.train() if training else model.eval()
    tot_t, tot_m, tot_p, n = 0.0, 0.0, 0.0, 0
    tc      = cfg["training"]
    warmup  = tc.get("lambda_warmup_epochs", 0)
    if training and epoch <= warmup:
        ltr, lpo, lhe = 0.0, 0.0, 0.0
    else:
        ltr = tc["lambda_trace"]
        lpo = tc["lambda_pos"]
        lhe = tc.get("lambda_herm", 0.02)
    use_amp = (scaler is not None) and (device.type == "cuda")
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp:
                with autocast("cuda"):
                    pred = model(x)
                    tot, mse, phys = compute_total_loss(pred, y, ltr, lpo, lhe)
                if training:
                    optimiser.zero_grad()
                    scaler.scale(tot).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
            else:
                pred = model(x)
                tot, mse, phys = compute_total_loss(pred, y, ltr, lpo, lhe)
                if training:
                    optimiser.zero_grad(); tot.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
            tot_t += tot.item(); tot_m += mse.item()
            tot_p += phys.item(); n += 1
    d = max(n, 1)
    return tot_t / d, tot_m / d, tot_p / d


def train_pinn(cfg):
    seed_everything(cfg["seed"])
    tc     = cfg["training"]
    device = get_device()
    warmup = tc.get("lambda_warmup_epochs", 0)
    print(f"\n{'='*60}\nTraining PINN\n{'='*60}")
    if warmup > 0:
        print(f"Physics warmup: {warmup} epochs  "
              f"λ_trace={tc['lambda_trace']} λ_pos={tc['lambda_pos']} "
              f"λ_herm={tc.get('lambda_herm', 0.02)}")

    dl_kw  = _dataloader_kwargs(cfg)
    tr_ld, va_ld, te_ld, norm, ti, vi, xi = build_dataloaders(cfg, "pinn", **dl_kw)
    model  = build_pinn_from_config(cfg).to(device)
    model  = maybe_compile(model, device)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"PINN: {n_p:,} parameters")

    opt    = Adam(model.parameters(), lr=tc["learning_rate"],
                  weight_decay=tc["weight_decay"])
    sch    = _make_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    os.makedirs(tc["checkpoint_dir"], exist_ok=True)
    os.makedirs(tc["log_dir"],        exist_ok=True)
    ckpt = os.path.join(tc["checkpoint_dir"], "pinn_best.pt")
    log  = CSVLogger(os.path.join(tc["log_dir"], "pinn_training.csv"),
                     ["epoch", "train_total", "train_mse", "train_phys",
                      "val_total", "val_mse", "lr", "elapsed_s"])

    best, pat, t0 = float("inf"), 0, time.perf_counter()
    for ep in range(1, tc["max_epochs"] + 1):
        tr_t, tr_m, tr_p = _pinn_epoch(model, tr_ld, device, cfg, scaler, opt, ep)
        va_t, va_m, va_p = _pinn_epoch(model, va_ld, device, cfg, epoch=ep)
        sch.step(va_t)
        lr = opt.param_groups[0]["lr"]
        log.log({"epoch": ep, "train_total": f"{tr_t:.6f}",
                 "train_mse": f"{tr_m:.6f}", "train_phys": f"{tr_p:.6f}",
                 "val_total": f"{va_t:.6f}", "val_mse": f"{va_m:.6f}",
                 "lr": f"{lr:.2e}", "elapsed_s": f"{time.perf_counter()-t0:.1f}"})
        if ep % 10 == 0 or ep == 1:
            tag = " [warmup]" if ep <= warmup else ""
            print(f"Ep {ep:4d} | tr[tot={tr_t:.5f} mse={tr_m:.5f} phys={tr_p:.5f}] "
                  f"val={va_t:.5f} lr={lr:.2e} [{time.perf_counter()-t0:.0f}s]{tag}")
        if va_t < best:
            best = va_t; pat = 0
            _save_ckpt(ckpt, model, norm, cfg, "pinn", ep, best, ti, vi, xi)
        else:
            pat += 1
            if pat >= tc["early_stopping_patience"]:
                print(f"Early stop at epoch {ep}"); break

    print(f"Best val loss: {best:.6f}  →  {ckpt}")
    ckpt_d = torch.load(ckpt, map_location=device)
    _load_state(model, ckpt_d)
    te_t, te_m, te_p = _pinn_epoch(model, te_ld, device, cfg)
    print(f"Test [total={te_t:.6f} mse={te_m:.6f} phys={te_p:.6f}]")
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Inverse flow
# ---------------------------------------------------------------------------

def _inv_epoch(model, loader, device, scaler=None, optimiser=None):
    training = optimiser is not None
    model.train() if training else model.eval()
    total, n = 0.0, 0
    use_amp = (scaler is not None) and (device.type == "cuda")
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for traj_flat, params_norm in loader:
            B       = traj_flat.shape[0]
            n_t     = traj_flat.shape[1] // N_RHO
            rho_seq = traj_flat.reshape(B, n_t, N_RHO).to(device, non_blocking=True)
            params  = params_norm.to(device, non_blocking=True)
            if use_amp:
                with autocast("cuda"):
                    log_p = model.log_prob(params, rho_seq)
                    loss  = -log_p.mean()
                if training:
                    optimiser.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
            else:
                log_p = model.log_prob(params, rho_seq)
                loss  = -log_p.mean()
                if training:
                    optimiser.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


def train_inverse(cfg):
    seed_everything(cfg["seed"])
    tc     = cfg["training"]
    device = get_device()
    print(f"\n{'='*60}\nTraining Inverse Flow\n{'='*60}")

    dl_kw  = _dataloader_kwargs(cfg)
    tr_ld, va_ld, te_ld, norm, ti, vi, xi = build_dataloaders(cfg, "inverse", **dl_kw)
    model  = build_inverse_from_config(cfg).to(device)
    model  = maybe_compile(model, device)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"Inverse flow: {n_p:,} parameters")

    opt    = Adam(model.parameters(), lr=tc["learning_rate"] * 0.3,
                  weight_decay=tc["weight_decay"])
    sch    = _make_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    os.makedirs(tc["checkpoint_dir"], exist_ok=True)
    os.makedirs(tc["log_dir"],        exist_ok=True)
    ckpt = os.path.join(tc["checkpoint_dir"], "inverse_best.pt")
    log  = CSVLogger(os.path.join(tc["log_dir"], "inverse_training.csv"),
                     ["epoch", "train_nll", "val_nll", "lr", "elapsed_s"])

    best, pat, t0 = float("inf"), 0, time.perf_counter()
    for ep in range(1, tc["max_epochs"] + 1):
        tr_l = _inv_epoch(model, tr_ld, device, scaler, opt)
        va_l = _inv_epoch(model, va_ld, device)
        sch.step(va_l)
        lr = opt.param_groups[0]["lr"]
        log.log({"epoch": ep, "train_nll": f"{tr_l:.6f}",
                 "val_nll": f"{va_l:.6f}", "lr": f"{lr:.2e}",
                 "elapsed_s": f"{time.perf_counter()-t0:.1f}"})
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:4d} | train_NLL={tr_l:.5f} val_NLL={va_l:.5f} "
                  f"lr={lr:.2e} [{time.perf_counter()-t0:.0f}s]")
        if va_l < best:
            best = va_l; pat = 0
            _save_ckpt(ckpt, model, norm, cfg, "inverse", ep, best, ti, vi, xi)
        else:
            pat += 1
            if pat >= tc["early_stopping_patience"]:
                print(f"Early stop at epoch {ep}"); break

    print(f"Best val NLL: {best:.6f}  →  {ckpt}")
    te_l = _inv_epoch(model, te_ld, device)
    print(f"Test NLL: {te_l:.6f}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Bootstrap ensemble
# ---------------------------------------------------------------------------

def train_ensemble(cfg):
    from uncertainty import BootstrapEnsemble
    device = get_device()
    print(f"\n{'='*60}\nTraining Bootstrap Ensemble\n{'='*60}")
    dl_kw  = _dataloader_kwargs(cfg)
    tr_ld, va_ld, _, norm, ti, vi, xi = build_dataloaders(cfg, "lstm", **dl_kw)
    ens = BootstrapEnsemble(cfg, device)
    ens.train(tr_ld, va_ld, verbose=True)
    ens_path = os.path.join(cfg["training"]["checkpoint_dir"], "ensemble.pkl")
    ens.save(ens_path)
    print(f"Ensemble saved → {ens_path}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")  # enable TF32 on Ampere/Ada/Hopper

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "pinn", "inverse", "ensemble", "all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    dispatch = {
        "lstm":     train_lstm,
        "pinn":     train_pinn,
        "inverse":  train_inverse,
        "ensemble": train_ensemble,
    }
    if args.model == "all":
        for name, fn in dispatch.items():
            print(f"\n{'#'*60}\n# {name.upper()}\n{'#'*60}")
            fn(cfg)
    else:
        dispatch[args.model](cfg)


if __name__ == "__main__":
    main()
