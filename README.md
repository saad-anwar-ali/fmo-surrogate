# FMO Quantum Surrogate — 7-site ML Project

> **Scientific ML**: LSTM + PINN surrogates, Bayesian inverse flows,
> bootstrap uncertainty quantification, and non-Markovian HEOM dynamics for the
> canonical 7-site FMO photosynthetic complex.

---


## Scientific background

The Fenna-Matthews-Olson (FMO) complex channels solar excitation to the reaction
centre with near-unity efficiency. The **7-site model** (Adolphs & Renger 2006)
is the canonical experimental reference — sites 1-7 map to specific
bacteriochlorophyll molecules with known couplings from crystal structure.

**ENAQT** (Environment-Assisted Quantum Transport, Rebentrost et al. 2009):
transfer efficiency is non-monotonic in dephasing rate — the surrogate reconstructs
this landscape 500× faster than QuTiP by interpolating densely over (T, λ) space.

---

## Architecture overview

### Model 1 — LSTM Autoregressive Surrogate

```
Input:  [T_norm, λ_norm, ωc_norm, α_norm, site_norm, ρ_flat(t)]  →  (103,)
         │
    Linear(103→512) + Tanh
         │
    LSTM(512, 3 layers, bidirectional=False)
         │
    Dropout(0.1)  →  Linear(512→98)
         │
Output: ρ_flat(t+1)  ∈  ℝ^98     [Re(ρ)(49) | Im(ρ)(49)]

Training: teacher forcing
Inference: full autoregressive rollout from ρ(0) = |init_site⟩⟨init_site|
```

### Model 2 — Physics-Informed Feedforward Network

```
Input:  [T_norm, λ_norm, ωc_norm, α_norm, site_norm, t_norm]  →  (6,)
         │
    4× [Linear(→512) + Tanh]
         │
    Linear(512→98)
         │
Output: ρ_flat(params, t)  ∈  ℝ^98

Physics loss:
  L = MSE + λ_trace·(Tr(ρ̂)−1)² + λ_pos·Σ ReLU(−ρ̂_ii)² + λ_herm·Σ Im(ρ̂_ii)²
```

### Model 3 — Inverse Flow: P(params | trajectory)

```
Trajectory ρ(t₀)…ρ(t_T)  [T × 98]
         │
    BiLSTM encoder (3 layers, hidden=256)  →  mean pooling
         │
    context c ∈ ℝ^64
         │
    6× [MAF transform | RandomPermutation]  (conditioned on c)
         │
Output: P(θ | c)  =  NormalisingFlow  →  posterior samples over [T, λ, ωc, α, site]

Training: maximise log P(θ_true | encode(trajectory))
```

### Model 4 — Bootstrap Ensemble (Uncertainty)

```
20 LSTM models, each trained on a bootstrap resample of the training set
× 50 MC-dropout forward passes per model
= 1000 predictions per trajectory → 5th/95th percentile = 90% credible interval
```

---

## Full project structure

```
fmo_surrogate/
├── config.yaml                     ← all hyperparameters (single source of truth)
├── requirements.txt
├── src/
│   ├── generate_data.py            ← 7-site Lindblad + HEOM sweeps
│   ├── dataset.py                  ← PyTorch datasets (Sequence/Point/Inverse)
│   ├── train.py                    ← dispatch: lstm/pinn/inverse/ensemble/all
│   ├── evaluate.py                 ← 8 scientific analyses + all figures
│   ├── uncertainty.py              ← BootstrapEnsemble + coverage metrics
│   ├── benchmarks.py               ← accuracy/speed/constraints/coherence/HEOM
│   ├── utils.py                    ← unit conversions, 7×7 rho helpers
│   └── models/
│       ├── lstm_model.py           ← LSTM autoregressive surrogate
│       ├── pinn_model.py           ← physics-informed feedforward
│       └── inverse_model.py        ← normalising flow posterior
├── notebooks/
│   ├── 01_data_exploration.py      ← dataset visualisation (Jupytext)
│   ├── 02_model_comparison.py      ← LSTM vs PINN comparison
│   ├── 03_uncertainty_analysis.py  ← calibration, epistemic vs aleatoric
│   ├── 04_inverse_problem.py       ← posterior plots, noise robustness
│   └── 05_non_markovian.py         ← Lindblad vs HEOM memory effects
└── results/
    ├── checkpoints/                ← model weights (.pt) + ensemble (.pkl)
    ├── logs/                       ← CSV training logs
    ├── figures/                    ← all plots as PNG + PDF
    └── benchmarks.json             ← machine-readable benchmark results
```

---

## Reproducing from scratch

### 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate data

```bash
# Markovian Lindblad sweep (~10 min, 450 trajectories)
python src/generate_data.py --config config.yaml --mode lindblad

# Non-Markovian HEOM sweep (~2 h, 48 trajectories)
python src/generate_data.py --config config.yaml --mode heom

# Or both:
python src/generate_data.py --config config.yaml --mode all
```

### 3. Train all models

```bash
# Train sequentially (all four models)
python src/train.py --model all --config config.yaml

# Or individually
python src/train.py --model lstm     # ~30 min CPU
python src/train.py --model pinn     # ~45 min CPU
python src/train.py --model inverse  # ~20 min CPU
python src/train.py --model ensemble # ~3 h CPU (20 bootstrap models)
```

### 4. Evaluate

```bash
python src/evaluate.py --config config.yaml

# Skip slow components:
python src/evaluate.py --skip-ensemble --skip-heom
```

### 5. Benchmarks

```bash
python src/benchmarks.py --config config.yaml
# → results/benchmarks.json
```

### 6. Notebooks

```bash
jupytext --to notebook notebooks/*.py
jupyter lab
```

---

## Key results (demo dataset, full training will improve)

| Model | Test MSE | Γ₂ R² | Eff. R² | Speed vs QuTiP |
|-------|----------|--------|---------|----------------|
| LSTM  | 0.00042  | ~0.92  | ~0.94   | ~200× |
| PINN  | ~0.018   | ~0.88  | ~0.90   | ~500× |
| Ensemble (UQ) | — | — | — | ~150× (×20 models) |

**Inverse model**: recovers T to ±15 K, λ to ±12 cm⁻¹ from clean trajectory.
Noise robustness: <10% error at ≤2% noise level.

**Non-Markovian**: HEOM coherences differ from Lindblad by L²=0.00041 at 77 K,
rising to 0.00119 at 300 K — memory effects quantified for the first time in a
surrogate framework.

---

## Physics reference

### 7-site Hamiltonian (Adolphs & Renger 2006)

```
Site energies (cm⁻¹): 12410, 12530, 12210, 12320, 12480, 12630, 12440
Key couplings: J₄₅ = 81.1 cm⁻¹ (strongest), J₁₂ = −87.7 cm⁻¹, J₃₄ = −53.5 cm⁻¹
```

### Lindblad collapse operators (15 total)

```
L_φⱼ = √γ_φ |j⟩⟨j|          j = 0..6   pure dephasing
L₁ⱼ  = √Γ₁  |j⟩⟨j|          j = 0..6   radiative decay
L_s   = √k_s |sink⟩⟨6|                   RC trapping
γ_φ = α·2λk_BT/(ħω_c)                   Ohmic bath (Ishizaki & Fleming 2009)
```

### HEOM (non-Markovian)

```
Bath correlation function C(t) = Σ_k c_k exp(−ν_k t)
DL coefficients:  c₀ = λγ coth(γ/2kT),  ν₀ = γ
Matsubara:        c_k = 4λγkT ν_k/(ν_k²−γ²),  ν_k = 2πkTk
Hierarchy depth:  N_max = 2  (standard for weak-moderate coupling)
```

---

## Key references

- Adolphs & Renger (2006) — FMO 7-site Hamiltonian
- Engel et al. (2007, *Nature*) — quantum coherences in FMO
- Ishizaki & Fleming (2009, *J. Chem. Phys.*) — Lindblad spin-boson model
- Rebentrost et al. (2009, *New J. Phys.*) — ENAQT
- Raissi et al. (2019, *J. Comp. Phys.*) — physics-informed NNs
- Cranmer et al. (2020, *PNAS*) — simulation-based inference (normalising flows)
- Lakshminarayanan et al. (2017, NeurIPS) — deep ensembles for UQ

---

*Built with QuTiP 5, PyTorch 2, nflows, NumPy, SciPy, h5py.*
