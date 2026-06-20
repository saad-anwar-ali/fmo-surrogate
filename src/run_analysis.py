"""
run_analysis.py  —  Master runner for all four new scientific analyses.

Usage
-----
    # Run everything
    python src/run_analysis.py --config config.yaml

    # Run individual analyses
    python src/run_analysis.py --config config.yaml --only enaqt
    python src/run_analysis.py --config config.yaml --only nonmarkovian
    python src/run_analysis.py --config config.yaml --only inverse
    python src/run_analysis.py --config config.yaml --only disorder

    # Disorder with custom N
    python src/run_analysis.py --config config.yaml --only disorder --N 500

Analyses
--------
  enaqt         — Physical ENAQT landscape extended to 5 ps (was 500 fs)
  nonmarkovian  — Lindblad–HEOM residual as vibronic fingerprint
  inverse       — Noisy recovery, identifiability, 2DES synthetic experiment
  disorder      — 1000-realisation static disorder ensemble + Jensen gap
"""

import argparse, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, seed_everything


def main():
    parser = argparse.ArgumentParser(
        description="Run scientific analyses for the FMO surrogate project.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only",   default=None,
                        choices=["enaqt", "nonmarkovian", "inverse", "disorder"],
                        help="Run only one analysis (default: all)")
    parser.add_argument("--N",      type=int, default=1000,
                        help="Disorder ensemble size (default: 1000)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg["evaluation"]["figure_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  FMO Surrogate — Scientific Analysis Suite")
    print(f"  Device: {device}  |  Save dir: {save_dir}")
    print(f"{'='*65}\n")

    run_all  = args.only is None
    t0_total = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Physical ENAQT landscape (5 ps)
    # ------------------------------------------------------------------
    if run_all or args.only == "enaqt":
        print("\n" + "="*65)
        print("  ANALYSIS 1: Physical ENAQT Landscape (5 ps)")
        print("="*65)
        t0 = time.perf_counter()
        try:
            from analysis.enaqt_physical import run_enaqt_physical_analysis
            run_enaqt_physical_analysis(cfg, device=device, save_dir=save_dir)
            print(f"  Done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 2. Non-Markovian / vibronic fingerprint
    # ------------------------------------------------------------------
    if run_all or args.only == "nonmarkovian":
        print("\n" + "="*65)
        print("  ANALYSIS 2: Non-Markovian Residual (Vibronic Fingerprint)")
        print("="*65)
        t0 = time.perf_counter()
        try:
            from analysis.non_markovian_residual import run_non_markovian_analysis
            run_non_markovian_analysis(cfg, device=device, save_dir=save_dir)
            print(f"  Done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 3. Inverse model applications
    # ------------------------------------------------------------------
    if run_all or args.only == "inverse":
        print("\n" + "="*65)
        print("  ANALYSIS 3: Inverse Model Applications")
        print("  (Noisy recovery · Identifiability · 2DES experiment)")
        print("="*65)
        t0 = time.perf_counter()
        try:
            from analysis.inverse_application import run_all_inverse_analyses
            run_all_inverse_analyses(cfg, device=device, save_dir=save_dir)
            print(f"  Done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. Static disorder ensemble
    # ------------------------------------------------------------------
    if run_all or args.only == "disorder":
        print("\n" + "="*65)
        print(f"  ANALYSIS 4: Static Disorder Ensemble (N={args.N})")
        print("="*65)
        t0 = time.perf_counter()
        try:
            from analysis.disorder_ensemble import run_all_disorder_analyses
            run_all_disorder_analyses(cfg, device=device, save_dir=save_dir,
                                       N_disorder=args.N)
            print(f"  Done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*65}")
    print(f"  All analyses complete in {time.perf_counter()-t0_total:.1f}s")
    print(f"  Figures saved to: {save_dir}/")
    print(f"{'='*65}\n")

    # Print figure inventory
    figs = sorted([f for f in os.listdir(save_dir)
                   if f.endswith((".png", ".pdf"))])
    if figs:
        print("  Generated figures:")
        for fig in figs:
            size_kb = os.path.getsize(os.path.join(save_dir, fig)) // 1024
            print(f"    {fig:50s}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
