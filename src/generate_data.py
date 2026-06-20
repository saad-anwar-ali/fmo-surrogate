"""
generate_data.py  —  FMO 7-site data generation (Lindblad + HEOM).
GPU-optimised version: parallel CPU workers for data generation
(QuTiP ODE solving is CPU-bound, but GPU machines have 16-32 cores
so we parallelise across trajectories using multiprocessing).

Key changes vs CPU version:
  - multiprocessing.Pool for Lindblad (embarrassingly parallel)
  - tqdm with parallel progress tracking
  - chunk-based HDF5 writing to avoid OOM for large sweeps
  - HEOM still serial (QuTiP HEOMSolver has internal parallelism)
  - auto-detects n_workers from CPU count

Usage
-----
    python src/generate_data.py --config config.yaml --mode lindblad
    python src/generate_data.py --config config.yaml --mode heom
    python src/generate_data.py --config config.yaml --mode all
    python src/generate_data.py --config config.yaml --mode lindblad --workers 16
"""

import argparse, itertools, json, os, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import h5py, numpy as np
import qutip as qt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (cm_to_rad_per_fs, dephasing_rate_cm, rho_matrix_to_flat,
                   extract_populations, extract_coherence_12, load_config,
                   seed_everything, K_B_CM_PER_K, N_SITES, N_RHO)

_QT_MAJOR = int(qt.__version__.split(".")[0])
_NDIM     = N_SITES + 1   # 8


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

def build_fmo_hamiltonian(cfg: dict) -> qt.Qobj:
    phys  = cfg["physics"]
    eps   = np.array(phys["site_energies_cm"], dtype=float)
    H_mat = np.zeros((_NDIM, _NDIM), dtype=complex)
    for j, e in enumerate(eps):
        H_mat[j, j] = cm_to_rad_per_fs(e)
    for (i, j, J_cm) in phys["couplings_cm"]:
        J_rf = cm_to_rad_per_fs(J_cm)
        H_mat[i, j] = J_rf
        H_mat[j, i] = J_rf
    return qt.Qobj(H_mat)


def _build_fmo_7only(cfg: dict) -> qt.Qobj:
    phys  = cfg["physics"]
    eps   = np.array(phys["site_energies_cm"], dtype=float)
    H_mat = np.diag([cm_to_rad_per_fs(e) for e in eps]).astype(complex)
    for (i, j, J_cm) in phys["couplings_cm"]:
        J_rf = cm_to_rad_per_fs(J_cm)
        H_mat[i, j] = J_rf
        H_mat[j, i] = J_rf
    return qt.Qobj(H_mat)


# ---------------------------------------------------------------------------
# Collapse operators
# ---------------------------------------------------------------------------

def build_collapse_operators(T_K, lambda_cm, omega_c_cm, alpha_scale, cfg) -> list:
    phys      = cfg["physics"]
    gp_rf     = cm_to_rad_per_fs(dephasing_rate_cm(T_K, lambda_cm, omega_c_cm, alpha_scale))
    Gamma1_rf = cm_to_rad_per_fs(phys["Gamma1_cm"])
    k_sink_rf = cm_to_rad_per_fs(phys["k_sink_cm"])
    b = [qt.basis(_NDIM, j) for j in range(_NDIM)]
    c_ops = []
    for j in range(N_SITES):
        proj = b[j] * b[j].dag()
        c_ops.append(np.sqrt(gp_rf)     * proj)
        c_ops.append(np.sqrt(Gamma1_rf) * proj)
    c_ops.append(np.sqrt(k_sink_rf) * b[N_SITES] * b[N_SITES - 1].dag())
    return c_ops


# ---------------------------------------------------------------------------
# Single Lindblad trajectory
# ---------------------------------------------------------------------------

def run_lindblad_trajectory(T_K, lambda_cm, omega_c_cm, alpha_scale,
                             init_site: int, H, t_list_fs, cfg) -> dict:
    rho0  = qt.basis(_NDIM, init_site) * qt.basis(_NDIM, init_site).dag()
    c_ops = build_collapse_operators(T_K, lambda_cm, omega_c_cm, alpha_scale, cfg)
    opts  = ({"nsteps": 10000, "rtol": 1e-8, "atol": 1e-10} if _QT_MAJOR >= 5
             else qt.Options(nsteps=10000, rtol=1e-8, atol=1e-10))
    result = qt.mesolve(H, rho0, t_list_fs, c_ops=c_ops, e_ops=[], options=opts)
    n     = len(t_list_fs)
    flat  = np.zeros((n, N_RHO), dtype=np.float32)
    sink  = np.zeros(n,          dtype=np.float32)
    b_snk = qt.basis(_NDIM, N_SITES)
    proj_s = b_snk * b_snk.dag()
    for i, rho_t in enumerate(result.states):
        full    = rho_t.full()
        flat[i] = rho_matrix_to_flat(full[:N_SITES, :N_SITES]).astype(np.float32)
        sink[i] = float(abs((proj_s * rho_t).tr()))
    return {
        "rho_flat":   flat,
        "coherence":  extract_coherence_12(flat),
        "pop":        extract_populations(flat),
        "sink_pop":   sink,
        "efficiency": float(np.clip(sink[-1], 0, 1)),
        "params":     np.array([T_K, lambda_cm, omega_c_cm, alpha_scale, init_site],
                               dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (must be top-level for pickle)
# ---------------------------------------------------------------------------

def _lindblad_worker(args):
    """Worker function: unpack args and run one trajectory."""
    T_K, lambda_cm, omega_c_cm, alpha_scale, init_site, cfg_dict, t_list_fs = args
    # Rebuild H inside worker (qt.Qobj not picklable across processes)
    from utils import load_config
    cfg = cfg_dict
    H   = build_fmo_hamiltonian(cfg)
    try:
        return run_lindblad_trajectory(
            T_K, lambda_cm, omega_c_cm, alpha_scale, int(init_site), H, t_list_fs, cfg)
    except Exception as e:
        print(f"\n  Worker failed T={T_K} λ={lambda_cm}: {e}")
        return None


# ---------------------------------------------------------------------------
# HEOM trajectory
# ---------------------------------------------------------------------------

def run_heom_trajectory(T_K, lambda_cm, omega_c_cm, alpha_scale,
                        init_site: int, H_fmo_only, t_list_fs, cfg) -> dict:
    from qutip.solver.heom.bofin_solvers import HEOMSolver
    from qutip.solver.heom.bofin_baths   import BosonicBath

    phys      = cfg["physics"]
    Nk        = phys["heom_Nk"]
    max_dep   = phys["heom_max_depth"]
    k_sink_rf = cm_to_rad_per_fs(phys["k_sink_cm"])

    lam_rf = cm_to_rad_per_fs(lambda_cm * alpha_scale)
    gam_rf = cm_to_rad_per_fs(omega_c_cm)
    kT_rf  = K_B_CM_PER_K * T_K * 2.0 * np.pi * 2.99792458e-5

    def dl_coefficients(lam, gamma, kT, Nk):
        coth  = 1.0 / np.tanh(gamma / (2.0 * kT))
        ck_r  = [lam * gamma * coth]
        vk_r  = [gamma]
        ck_i  = [-lam * gamma]
        vk_i  = [gamma]
        for k in range(1, Nk + 1):
            nu_k = 2.0 * np.pi * kT * k
            denom = nu_k**2 - gamma**2
            if abs(denom) < 1e-20:
                continue
            c_k = 4.0 * lam * gamma * kT * nu_k / denom
            ck_r.append(c_k)
            vk_r.append(nu_k)
        return ck_r, vk_r, ck_i, vk_i

    baths = []
    for j in range(N_SITES):
        Q_j   = qt.Qobj(np.diag([1.0 if k == j else 0.0 for k in range(N_SITES)]))
        ck_r, vk_r, ck_i, vk_i = dl_coefficients(lam_rf, gam_rf, kT_rf, Nk)
        baths.append(BosonicBath(Q_j, ck_r, vk_r, ck_i, vk_i))

    b7 = [qt.basis(N_SITES, j) for j in range(N_SITES)]
    sink_proj = b7[N_SITES - 1] * b7[N_SITES - 1].dag()
    H_eff = H_fmo_only - 0.5j * k_sink_rf * sink_proj

    rho0   = qt.basis(N_SITES, init_site) * qt.basis(N_SITES, init_site).dag()
    solver = HEOMSolver(H_eff, baths, max_depth=max_dep,
                        options={"nsteps": 10000, "rtol": 1e-8, "atol": 1e-10,
                                 "progress_bar": False})
    result = solver.run(rho0, t_list_fs, e_ops=[])

    n      = len(t_list_fs)
    flat   = np.zeros((n, N_RHO), dtype=np.float32)
    traces = np.zeros(n, dtype=np.float32)
    for i, rho_t in enumerate(result.states):
        flat[i]   = rho_matrix_to_flat(rho_t.full()).astype(np.float32)
        traces[i] = float(rho_t.tr().real)

    sink_pop = np.clip(1.0 - traces, 0, 1)
    return {
        "rho_flat":   flat,
        "coherence":  extract_coherence_12(flat),
        "pop":        extract_populations(flat),
        "sink_pop":   sink_pop,
        "efficiency": float(sink_pop[-1]),
        "params":     np.array([T_K, lambda_cm, omega_c_cm, alpha_scale, init_site],
                               dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# HDF5 writer (chunk-based for large datasets)
# ---------------------------------------------------------------------------

def write_hdf5(path: str, trajectories: list, t_fs: np.ndarray,
               cfg: dict, dataset_label: str = "lindblad") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n_traj  = len(trajectories)
    n_steps = len(t_fs)
    n_rho   = trajectories[0]["rho_flat"].shape[1]
    print(f"\nWriting {n_traj} {dataset_label} trajectories → {path}")
    with h5py.File(path, "w") as f:
        tg = f.create_group("trajectories")
        tg.create_dataset("rho_flat",   shape=(n_traj, n_steps, n_rho), dtype=np.float32,
                          chunks=(min(64, n_traj), n_steps, n_rho),
                          compression="gzip", compression_opts=4)
        tg.create_dataset("params",     shape=(n_traj, 5),              dtype=np.float32)
        tg.create_dataset("coherence",  shape=(n_traj, n_steps),        dtype=np.float32)
        tg.create_dataset("pop",        shape=(n_traj, n_steps, N_SITES),dtype=np.float32)
        tg.create_dataset("sink_pop",   shape=(n_traj, n_steps),        dtype=np.float32)
        tg.create_dataset("efficiency", shape=(n_traj,),                 dtype=np.float32)
        tg.create_dataset("t_fs",       data=t_fs.astype(np.float32))
        for i, t in enumerate(trajectories):
            tg["rho_flat"][i]   = t["rho_flat"]
            tg["params"][i]     = t["params"]
            tg["coherence"][i]  = t["coherence"]
            tg["pop"][i]        = t["pop"]
            tg["sink_pop"][i]   = t["sink_pop"]
            tg["efficiency"][i] = t["efficiency"]
        tg["params"].attrs["columns"] = ["T_K","lambda_cm","omega_c_cm","alpha_scale","init_site"]
        mg = f.create_group("metadata")
        dt = h5py.special_dtype(vlen=str)
        pn = mg.create_dataset("param_names", (5,), dtype=dt)
        pn[:] = ["T_K","lambda_cm","omega_c_cm","alpha_scale","init_site"]
        mg.create_dataset("dataset_type", data=dataset_label)
        cd = mg.create_dataset("config", (1,), dtype=dt)
        cd[0] = json.dumps(cfg, indent=2)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Done. {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Lindblad sweep — parallel
# ---------------------------------------------------------------------------

def run_lindblad_sweep(cfg: dict, n_workers: int = None) -> None:
    seed_everything(cfg["seed"])
    sw   = cfg["sweep"]
    t_fs = np.linspace(sw["t_start_fs"], sw["t_end_fs"], sw["n_steps"])

    combos = list(itertools.product(
        sw["T_K"], sw["lambda_cm"], sw["omega_c_cm"],
        sw["alpha_scale"], sw["initial_sites"]))
    print(f"Lindblad sweep: {len(combos)} trajectories")
    print(f"  t_end={sw['t_end_fs']:.0f} fs, n_steps={sw['n_steps']}, "
          f"Δt={t_fs[1]-t_fs[0]:.1f} fs")

    # Auto-detect workers
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    print(f"  Using {n_workers} parallel workers")

    # Time estimate from single trajectory
    H_test = build_fmo_hamiltonian(cfg)
    t0 = time.perf_counter()
    run_lindblad_trajectory(300.0, 55.0, 100.0, 1.0, 0, H_test, t_fs[:50], cfg)
    t_sample = time.perf_counter() - t0
    t_single_est = t_sample * len(t_fs) / 50
    print(f"  Single traj est: {t_single_est:.2f}s | "
          f"Parallel est: {t_single_est*len(combos)/n_workers/60:.1f} min "
          f"({n_workers} workers)")

    # Build worker args (cfg as dict — picklable)
    worker_args = [
        (T, lam, omc, alp, int(site), cfg, t_fs)
        for T, lam, omc, alp, site in combos
    ]

    trajs = []
    failed = 0
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(_lindblad_worker, worker_args, chunksize=4),
            total=len(combos), desc="Lindblad parallel", unit="traj"
        ):
            if result is not None:
                trajs.append(result)
            else:
                failed += 1

    if failed > 0:
        print(f"  WARNING: {failed} trajectories failed and were skipped.")
    print(f"  Generated {len(trajs)} trajectories successfully.")

    write_hdf5(cfg["data"]["lindblad_hdf5"], trajs, t_fs, cfg, "lindblad")
    effs = np.array([t["efficiency"] for t in trajs])
    print(f"Efficiency: mean={effs.mean():.4f}  std={effs.std():.4f}  "
          f"min={effs.min():.4f}  max={effs.max():.4f}")


# ---------------------------------------------------------------------------
# HEOM sweep — serial (internal parallelism in HEOMSolver)
# ---------------------------------------------------------------------------

def run_heom_sweep(cfg: dict, n_workers: int = None) -> None:
    seed_everything(cfg["seed"])
    sw   = cfg["sweep"]
    H_7  = _build_fmo_7only(cfg)
    t_fs = np.linspace(sw["t_start_fs"], sw["heom_t_end_fs"], sw["heom_n_steps"])

    combos = list(itertools.product(
        sw["heom_T_K"], sw["heom_lambda_cm"], sw["heom_omega_c_cm"],
        sw["heom_alpha_scale"], sw["initial_sites"]))
    print(f"\nHEOM sweep: {len(combos)} trajectories "
          f"(Nk={cfg['physics']['heom_Nk']}, "
          f"max_depth={cfg['physics']['heom_max_depth']})")
    print(f"  t_end={sw['heom_t_end_fs']:.0f} fs, n_steps={sw['heom_n_steps']}")

    # Time estimate
    t0 = time.perf_counter()
    try:
        run_heom_trajectory(200.0, 55.0, 100.0, 1.0, 0, H_7, t_fs[:10], cfg)
        t_s = time.perf_counter() - t0
        print(f"  10-step HEOM sample: {t_s:.2f}s | "
              f"Est. total: {t_s*len(t_fs)/10*len(combos)/60:.1f} min (serial)")
        print(f"  TIP: Run 'python src/generate_data.py --mode heom' in parallel "
              f"across {min(len(combos), 8)} terminal sessions for speedup")
    except Exception as e:
        print(f"  HEOM test failed: {e}")
        return

    trajs = []
    with tqdm(total=len(combos), desc="HEOM 7-site", unit="traj") as pbar:
        for T, lam, omc, alp, site in combos:
            try:
                result = run_heom_trajectory(
                    T, lam, omc, alp, int(site), H_7, t_fs, cfg)
                trajs.append(result)
                pbar.set_postfix(T=f"{T:.0f}K", eff=f"{result['efficiency']:.4f}")
            except Exception as e:
                print(f"\n  HEOM failed T={T} λ={lam}: {e}. Skipping.")
            pbar.update(1)

    if trajs:
        write_hdf5(cfg["data"]["heom_hdf5"], trajs, t_fs, cfg, "heom")
        effs = np.array([t["efficiency"] for t in trajs])
        print(f"HEOM Efficiency: mean={effs.mean():.4f}  "
              f"min={effs.min():.4f}  max={effs.max():.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FMO 7-site dataset.")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--mode",    default="lindblad",
                        choices=["lindblad", "heom", "all"])
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for Lindblad "
                             "(default: auto = min(cpu_count, 16))")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    if args.mode in ("lindblad", "all"):
        run_lindblad_sweep(cfg, n_workers=args.workers)
    if args.mode in ("heom", "all"):
        run_heom_sweep(cfg)
