"""
benchmarks.py  —  Comprehensive benchmark suite for the FMO surrogate project.

Benchmarks reported:
  1. Accuracy:      MSE, MAE, R²(coherence), R²(efficiency) — LSTM vs PINN
  2. Speed:         Wall-clock per trajectory — QuTiP vs LSTM vs PINN
                    FIX: both use identical timing protocol (5 reps, drop first)
                    FIX: PINN correctly benchmarked as a single batched forward pass
  3. Constraints:   Trace error, positivity violation rate, Hermiticity error
  4. Coherence:     Γ₂ R² using Hilbert envelope (FIX from evaluate.py)
  5. Non-Markovian: Lindblad vs HEOM trajectory divergence
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_config, seed_everything, extract_populations,
                   extract_coherence_12, compute_trace, N_RHO, N_SITES)
from dataset import ParamNormaliser


def _load_ckpt(name, cfg, device):
    from models.lstm_model import build_lstm_from_config
    from models.pinn_model import build_pinn_from_config
    path = os.path.join(cfg["training"]["checkpoint_dir"], f"{name}_best.pt")
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location=device)
    m = (build_lstm_from_config(cfg) if name == "lstm"
         else build_pinn_from_config(cfg)).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, ParamNormaliser.from_dict(ckpt["normaliser"])


@torch.no_grad()
def _predict_lstm(model, params_norm, rho0, n_steps, device):
    p = torch.tensor(params_norm, dtype=torch.float32, device=device).unsqueeze(0)
    r = torch.tensor(rho0,        dtype=torch.float32, device=device).unsqueeze(0)
    return model.rollout(p, r, n_steps).squeeze(0).cpu().numpy()

@torch.no_grad()
def _predict_pinn(model, params_norm, t_norm, device):
    n  = len(t_norm)
    pn = np.tile(params_norm, (n, 1)).astype(np.float32)
    t  = t_norm.reshape(-1, 1).astype(np.float32)
    x  = torch.tensor(np.concatenate([pn, t], axis=-1), dtype=torch.float32, device=device)
    return model(x).cpu().numpy()

def _get_test_idx(cfg):
    from sklearn.model_selection import train_test_split
    import h5py
    with h5py.File(cfg["data"]["lindblad_hdf5"], "r") as f:
        n_traj = f["trajectories/params"].shape[0]
    idx = np.arange(n_traj)
    vt  = cfg["split"]["val_frac"] + cfg["split"]["test_frac"]
    _, vt_idx = train_test_split(idx, test_size=vt, random_state=cfg["seed"])
    tf = cfg["split"]["test_frac"] / vt
    _, test_idx = train_test_split(vt_idx, test_size=tf, random_state=cfg["seed"])
    return test_idx

def _fit_gamma2(t_fs, coh):
    """Fit exponential to Hilbert envelope — same fix as evaluate.py."""
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


# ---------------------------------------------------------------------------
# 1. Accuracy
# ---------------------------------------------------------------------------

def benchmark_accuracy(cfg, device):
    hdf5   = cfg["data"]["lindblad_hdf5"]
    lstm_m, lstm_n = _load_ckpt("lstm", cfg, device)
    pinn_m, pinn_n = _load_ckpt("pinn", cfg, device)
    test_idx = _get_test_idx(cfg)

    with h5py.File(hdf5, "r") as f:
        t_fs = f["trajectories/t_fs"][:]
    t_norm  = (t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())
    n_steps = len(t_fs)

    res = {}
    for model, norm, key in [(lstm_m, lstm_n, "lstm"), (pinn_m, pinn_n, "pinn")]:
        if model is None: continue
        mses, maes, coh_true, coh_pred, eff_true, eff_pred = [], [], [], [], [], []
        with h5py.File(hdf5, "r") as f:
            for i in test_idx:
                gt  = f["trajectories/rho_flat"][i]
                par = f["trajectories/params"][i]
                eff_true.append(float(f["trajectories/efficiency"][i]))
                pn   = norm.transform(par.astype(np.float32))
                pred = (_predict_lstm(model, pn, gt[0], n_steps, device) if key == "lstm"
                        else _predict_pinn(model, pn, t_norm, device))
                mses.append(float(np.mean((pred - gt)**2)))
                maes.append(float(np.mean(np.abs(pred - gt))))
                coh_true.append(extract_coherence_12(gt).mean())
                coh_pred.append(extract_coherence_12(pred).mean())
                # Efficiency: for LSTM use 1-Tr(rho_end), for PINN use site-7 integral
                if key == "lstm":
                    tr_f = float(np.sum(pred[-1, :N_SITES*N_SITES].reshape(N_SITES, N_SITES).diagonal()))
                    eff_pred.append(float(np.clip(1.0 - tr_f, 0, 1)))
                else:
                    pops_7 = extract_populations(pred)[:, 6]
                    eff_pred.append(float(np.clip(np.trapz(pops_7, t_norm) * 0.05, 0, 1)))

        res[key] = {
            "mse": float(np.mean(mses)),
            "mae": float(np.mean(maes)),
            "r2_coherence": r2_score(coh_true, coh_pred),
            "r2_efficiency": r2_score(eff_true, eff_pred),
            "n_test": len(test_idx),
        }
    return res


# ---------------------------------------------------------------------------
# 2. Speed  (FIX: identical protocol for all models)
# ---------------------------------------------------------------------------

def benchmark_speed(cfg, device):
    """
    All three solvers are timed on the same parameter point (T=300K, λ=55,
    ωc=100, α=1, site 1) with identical protocol:
      - 10 repetitions, first 2 dropped (warmup)
      - Mean of remaining 8 reported

    PINN speedup was previously wrong (showed 0.56×, i.e. *slower* than QuTiP)
    because the time loop inside predict_pinn was not the bottleneck — the
    bottleneck was Python overhead from calling it 200 times per trajectory in
    the benchmark loop.  predict_pinn already batches all 200 time points in
    one forward pass, which is the correct comparison.  The fix ensures the
    QuTiP timer includes full trajectory integration (200 steps), not just
    the solver setup.
    """
    import qutip as qt
    from generate_data import build_fmo_hamiltonian, run_lindblad_trajectory

    lstm_m, lstm_n = _load_ckpt("lstm", cfg, device)
    pinn_m, pinn_n = _load_ckpt("pinn", cfg, device)

    sw     = cfg["sweep"]
    t_fs   = np.linspace(sw["t_start_fs"], sw["t_end_fs"], sw["n_steps"])
    t_norm = (t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())
    H      = build_fmo_hamiltonian(cfg)
    n_steps = len(t_fs)

    dummy_raw = np.array([300.0, 55.0, 100.0, 1.0, 0.0], dtype=np.float32)
    rho0 = np.zeros(N_RHO, dtype=np.float32); rho0[0] = 1.0

    REPS = 10
    WARMUP = 2

    def _time_fn(fn, reps=REPS, warmup=WARMUP):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.mean(times[warmup:]))

    t_qt   = _time_fn(lambda: run_lindblad_trajectory(
        300.0, 55.0, 100.0, 1.0, 0, H, t_fs, cfg))

    t_lstm = None
    if lstm_m:
        pn_l = lstm_n.transform(dummy_raw)
        t_lstm = _time_fn(lambda: _predict_lstm(lstm_m, pn_l, rho0, n_steps, device))

    t_pinn = None
    if pinn_m:
        pn_p = pinn_n.transform(dummy_raw)
        # One batched forward pass over all 200 time points — the correct
        # comparison since this is how the PINN is used in practice.
        t_pinn = _time_fn(lambda: _predict_pinn(pinn_m, pn_p, t_norm, device))

    su_lstm = t_qt / t_lstm if t_lstm else None
    su_pinn = t_qt / t_pinn if t_pinn else None

    print(f"\nSpeed benchmark (averaged over {REPS-WARMUP} reps):")
    print(f"  QuTiP: {t_qt*1000:.1f} ms")
    if t_lstm: print(f"  LSTM:  {t_lstm*1000:.2f} ms  (×{su_lstm:.2f} vs QuTiP)")
    if t_pinn: print(f"  PINN:  {t_pinn*1000:.2f} ms  (×{su_pinn:.2f} vs QuTiP)")

    return {"qutip_s": t_qt, "lstm_s": t_lstm, "pinn_s": t_pinn,
            "speedup_lstm": su_lstm, "speedup_pinn": su_pinn}


# ---------------------------------------------------------------------------
# 3. Physical constraints
# ---------------------------------------------------------------------------

def benchmark_constraints(cfg, device):
    hdf5   = cfg["data"]["lindblad_hdf5"]
    lstm_m, lstm_n = _load_ckpt("lstm", cfg, device)
    pinn_m, pinn_n = _load_ckpt("pinn", cfg, device)
    test_idx = _get_test_idx(cfg)

    with h5py.File(hdf5, "r") as f:
        t_fs = f["trajectories/t_fs"][:]
    t_norm  = (t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())
    n_steps = len(t_fs)

    res = {}
    for model, norm, key in [(lstm_m, lstm_n, "lstm"), (pinn_m, pinn_n, "pinn")]:
        if model is None: continue
        trace_errs, pos_viols, herm_viols = [], [], []
        with h5py.File(hdf5, "r") as f:
            for i in test_idx:
                gt  = f["trajectories/rho_flat"][i]
                par = f["trajectories/params"][i]
                pn  = norm.transform(par.astype(np.float32))
                pred = (_predict_lstm(model, pn, gt[0], n_steps, device) if key == "lstm"
                        else _predict_pinn(model, pn, t_norm, device))
                # Trace: sum of diagonal of Re(rho) at t_final
                diag = pred[-1, :N_SITES*N_SITES].reshape(N_SITES, N_SITES).diagonal()
                trace_pred = float(np.sum(diag))
                diag_gt = gt[-1, :N_SITES*N_SITES].reshape(N_SITES, N_SITES).diagonal()
                trace_gt = float(np.sum(diag_gt))
                trace_errs.append(abs(trace_pred - trace_gt))
                # Positivity
                pops = extract_populations(pred)
                pos_viols.append(float(np.any(pops < -0.005)))
                # Hermiticity: Im(diag) should be 0
                im_diag_idx = N_SITES*N_SITES + np.arange(N_SITES)*(N_SITES+1)
                herm_viols.append(float(np.mean(np.abs(pred[:, im_diag_idx]))))

        res[key] = {
            "mean_trace_error":          float(np.mean(trace_errs)),
            "positivity_violation_rate": float(np.mean(pos_viols)),
            "mean_hermiticity_error":    float(np.mean(herm_viols)),
        }
    return res


# ---------------------------------------------------------------------------
# 4. Coherence decay (Hilbert envelope — same fix as evaluate.py)
# ---------------------------------------------------------------------------

def benchmark_coherence(cfg, device):
    hdf5   = cfg["data"]["lindblad_hdf5"]
    lstm_m, lstm_n = _load_ckpt("lstm", cfg, device)
    pinn_m, pinn_n = _load_ckpt("pinn", cfg, device)
    test_idx = _get_test_idx(cfg)

    with h5py.File(hdf5, "r") as f:
        t_fs = f["trajectories/t_fs"][:]
    t_norm  = (t_fs - t_fs.min()) / (t_fs.max() - t_fs.min())
    n_steps = len(t_fs)

    res = {}
    for model, norm, key in [(lstm_m, lstm_n, "lstm"), (pinn_m, pinn_n, "pinn")]:
        if model is None: continue
        gt_rates, pred_rates, T_vals = [], [], []
        with h5py.File(hdf5, "r") as f:
            for i in test_idx:
                gt  = f["trajectories/rho_flat"][i]
                par = f["trajectories/params"][i]
                pn  = norm.transform(par.astype(np.float32))
                pred = (_predict_lstm(model, pn, gt[0], n_steps, device) if key == "lstm"
                        else _predict_pinn(model, pn, t_norm, device))
                r_gt,   ok1 = _fit_gamma2(t_fs, extract_coherence_12(gt))
                r_pred, ok2 = _fit_gamma2(t_fs, extract_coherence_12(pred))
                if ok1 and ok2 and not (np.isnan(r_gt) or np.isnan(r_pred)):
                    gt_rates.append(r_gt); pred_rates.append(r_pred)
                    T_vals.append(float(par[0]))

        if len(gt_rates) > 1:
            r2 = r2_score(gt_rates, pred_rates)
            c_gt = np.polyfit(T_vals, gt_rates, 1)
            c_p  = np.polyfit(T_vals, pred_rates, 1)
            res[key] = {"r2_gamma2": r2,
                        "gt_slope_per_K":   float(c_gt[0]),
                        "pred_slope_per_K": float(c_p[0]),
                        "slope_recovery_pct": float(
                            100*(1 - abs(c_p[0]-c_gt[0])/max(abs(c_gt[0]), 1e-20)))}
        else:
            res[key] = {"r2_gamma2": float("nan")}
    return res


# ---------------------------------------------------------------------------
# 5. Non-Markovian divergence
# ---------------------------------------------------------------------------

def benchmark_non_markovian(cfg):
    lpath = cfg["data"]["lindblad_hdf5"]
    hpath = cfg["data"]["heom_hdf5"]
    if not os.path.exists(hpath):
        return {"status": "heom_dataset_not_found"}

    with h5py.File(lpath, "r") as fl, h5py.File(hpath, "r") as fh:
        l_params = fl["trajectories/params"][:]
        h_params = fh["trajectories/params"][:]
        h_t_fs   = fh["trajectories/t_fs"][:]

    divergences_by_T = {}
    for T in np.unique(h_params[:, 0]):
        divs = []
        with h5py.File(lpath, "r") as fl, h5py.File(hpath, "r") as fh:
            for lam in np.unique(h_params[h_params[:, 0] == T, 1]):
                lm = np.where((l_params[:, 0] == T) & (l_params[:, 1] == lam))[0]
                hm = np.where((h_params[:, 0] == T) & (h_params[:, 1] == lam))[0]
                if len(lm) == 0 or len(hm) == 0: continue
                rho_l = fl["trajectories/rho_flat"][lm[0]]
                rho_h = fh["trajectories/rho_flat"][hm[0]]
                min_s = min(rho_l.shape[0], rho_h.shape[0])
                divs.append(np.mean((rho_l[:min_s] - rho_h[:min_s])**2, axis=1))
        if divs:
            divs = np.array(divs)
            divergences_by_T[str(T)] = {
                "mean_l2_divergence": float(divs.mean()),
                "max_l2_divergence":  float(divs.max()),
                "peak_time_fs":       float(h_t_fs[np.argmax(divs.mean(axis=0))]),
            }
    return {"per_temperature": divergences_by_T, "status": "ok",
            "note": "L2 distance between Lindblad and HEOM density matrices. Larger at low T."}


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_benchmarks(cfg, device):
    results = {}

    print("\n" + "="*60 + "\nBENCHMARK 1: Accuracy\n" + "="*60)
    try:
        results["accuracy"] = benchmark_accuracy(cfg, device)
        for k, v in results["accuracy"].items():
            print(f"  {k}: MSE={v.get('mse','-'):.6f}  MAE={v.get('mae','-'):.6f}  "
                  f"R²_coh={v.get('r2_coherence','-'):.4f}  R²_eff={v.get('r2_efficiency','-'):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}"); results["accuracy"] = {"error": str(e)}

    print("\n" + "="*60 + "\nBENCHMARK 2: Speed\n" + "="*60)
    try:
        results["speed"] = benchmark_speed(cfg, device)
    except Exception as e:
        print(f"  ERROR: {e}"); results["speed"] = {"error": str(e)}

    print("\n" + "="*60 + "\nBENCHMARK 3: Physical Constraints\n" + "="*60)
    try:
        results["constraints"] = benchmark_constraints(cfg, device)
        for k, v in results["constraints"].items():
            print(f"  {k}: trace_err={v.get('mean_trace_error','-'):.5f}  "
                  f"pos_viol={v.get('positivity_violation_rate','-'):.3f}")
    except Exception as e:
        print(f"  ERROR: {e}"); results["constraints"] = {"error": str(e)}

    print("\n" + "="*60 + "\nBENCHMARK 4: Coherence Decay Rates\n" + "="*60)
    try:
        results["coherence"] = benchmark_coherence(cfg, device)
        for k, v in results["coherence"].items():
            print(f"  {k}: R²(Γ₂)={v.get('r2_gamma2','-'):.4f}  "
                  f"slope_recovery={v.get('slope_recovery_pct','-'):.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}"); results["coherence"] = {"error": str(e)}

    print("\n" + "="*60 + "\nBENCHMARK 5: Non-Markovian Divergence\n" + "="*60)
    try:
        results["non_markovian"] = benchmark_non_markovian(cfg)
        if results["non_markovian"].get("status") == "ok":
            for T, v in results["non_markovian"]["per_temperature"].items():
                print(f"  T={T}K: mean_L2={v['mean_l2_divergence']:.4f}  "
                      f"peak@{v['peak_time_fs']:.0f}fs")
    except Exception as e:
        print(f"  ERROR: {e}"); results["non_markovian"] = {"error": str(e)}

    bench_path = cfg["evaluation"]["benchmark_file"]
    os.makedirs(os.path.dirname(os.path.abspath(bench_path)), exist_ok=True)
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmarks saved to {bench_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    seed_everything(cfg["seed"])
    run_all_benchmarks(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
