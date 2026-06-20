"""
inverse_model.py  —  Bayesian inverse problem via normalising flows.

Goal: given a noisy density matrix trajectory rho(t), infer the posterior
distribution over the physical parameters:

    P(theta | rho(t))  ∝  P(rho(t) | theta) * P(theta)

where theta = [T, lambda, omega_c, alpha, init_site].

Architecture
------------
1. Trajectory encoder (BiLSTM + mean-pooling):
   rho(t₀), ..., rho(t_T)  →  context vector c ∈ R^{encoder_out}

2. Conditional Masked Autoregressive Flow (MAF):
   P(theta | c) = NormalisingFlow(base=N(0,I), transforms=MAF(context=c))

   The flow learns an invertible map z = f(theta; c) such that:
     log P(theta | c) = log P_base(f(theta;c)) + log|det J_f|

   Sampling: theta ~ P(theta | c)  by  theta = f^{-1}(z; c),  z ~ N(0,I)

Why normalising flows?
----------------------
• Full posterior (not just a point estimate) — critical for scientific credibility
• Exact log-likelihood computation — enables model comparison (BIC/AIC analogues)
• Samples allow uncertainty propagation downstream (e.g. into ENAQT landscape)
• Novel for FMO parameter estimation — state of the art for simulation-based inference

Reference: Cranmer, Brehmer & Louppe (2020) "The frontier of simulation-based inference"

Training objective
------------------
Maximise log P(theta_true | c):
    L = -E_{(trajectory, theta)} [ log P(theta | encode(trajectory)) ]

This is the negative log-likelihood of the true parameters under the
flow-predicted posterior conditioned on the encoded trajectory.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# nflows: Normalising Flows library
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    RandomPermutation,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import N_RHO, N_SITES


# ---------------------------------------------------------------------------
# Trajectory encoder
# ---------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    """
    Encode a variable-length rho(t) time series into a fixed-size context vector.

    Uses a bidirectional LSTM so that each time step has access to both
    past and future context. The final context is the mean over all hidden
    states (mean-pooling is more stable than last-step for variable lengths).

    Input:  (B, T, N_RHO)  —  density matrix sequence
    Output: (B, encoder_out)  —  context vector
    """

    def __init__(self, n_rho: int = N_RHO,
                 hidden_size: int = 256,
                 n_layers: int = 3,
                 encoder_out: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_rho, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # BiLSTM output is 2*hidden (forward + backward)
        self.projector = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, encoder_out),
        )

    def forward(self, rho_seq: Tensor) -> Tensor:
        """
        Parameters
        ----------
        rho_seq : Tensor, shape (B, T, N_RHO)
            Density matrix time series (possibly noisy for inverse task).

        Returns
        -------
        Tensor, shape (B, encoder_out)
            Fixed-size context vector summarising the trajectory.
        """
        x   = self.input_proj(rho_seq)          # (B, T, H)
        out, _ = self.bilstm(x)                 # (B, T, 2H)
        ctx = out.mean(dim=1)                    # (B, 2H)  — mean pooling
        return self.projector(ctx)               # (B, encoder_out)


# ---------------------------------------------------------------------------
# Conditional normalising flow
# ---------------------------------------------------------------------------

def build_conditional_flow(n_params: int = 5,
                           context_dim: int = 64,
                           hidden_features: int = 128,
                           n_transforms: int = 6) -> Flow:
    """
    Build a conditional Masked Autoregressive Flow (MAF).

    Each MAF layer learns an autoregressive affine transform of the
    parameters conditioned on the encoder context:
        s_i, t_i = NN(theta_{<i}; context)
        z_i = theta_i * exp(s_i) + t_i

    Alternating permutations ensure all dimensions are mixed.

    Parameters
    ----------
    n_params : int
        Dimensionality of the parameter space (5 for FMO).
    context_dim : int
        Dimensionality of the encoder context vector.
    hidden_features : int
        Width of the MAF conditioner networks.
    n_transforms : int
        Number of alternating MAF + permutation layers.

    Returns
    -------
    nflows.flows.Flow
        Conditional normalising flow.
    """
    transforms = []
    for _ in range(n_transforms):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=n_params,
                hidden_features=hidden_features,
                context_features=context_dim,
                num_blocks=2,
                use_residual_blocks=True,
                activation=F.tanh,
            )
        )
        transforms.append(RandomPermutation(features=n_params))

    base = StandardNormal([n_params])
    flow = Flow(CompositeTransform(transforms), base)
    return flow


# ---------------------------------------------------------------------------
# Full inverse model
# ---------------------------------------------------------------------------

class InverseFlowModel(nn.Module):
    """
    Full inverse surrogate: trajectory → P(params | trajectory).

    Combines the trajectory encoder and the conditional normalising flow
    into a single nn.Module for clean checkpointing and inference.

    Training
    --------
    Minimise NLL:  L = -mean_batch [ log P_flow(params_true | encode(traj)) ]

    Inference
    ---------
    Given a new (possibly noisy) trajectory:
      1. context = encode(trajectory)
      2. Sample N parameter vectors from P_flow(params | context)
      3. Transform samples back to physical units via ParamNormaliser

    The result is a full posterior distribution over the physical parameters.
    """

    def __init__(self,
                 n_rho:          int = N_RHO,
                 n_params:       int = 5,
                 encoder_hidden: int = 256,
                 encoder_layers: int = 3,
                 encoder_out:    int = 64,
                 flow_hidden:    int = 128,
                 flow_layers:    int = 6,
                 flow_dropout:   float = 0.05):
        super().__init__()

        self.n_params    = n_params
        self.encoder_out = encoder_out

        self.encoder = TrajectoryEncoder(
            n_rho       = n_rho,
            hidden_size = encoder_hidden,
            n_layers    = encoder_layers,
            encoder_out = encoder_out,
            dropout     = flow_dropout,
        )
        self.flow = build_conditional_flow(
            n_params        = n_params,
            context_dim     = encoder_out,
            hidden_features = flow_hidden,
            n_transforms    = flow_layers,
        )

    def log_prob(self, params_norm: Tensor, rho_seq: Tensor) -> Tensor:
        """
        Compute log P(params_norm | rho_seq) for a batch.

        Parameters
        ----------
        params_norm : Tensor, shape (B, n_params)
            Normalised ground-truth physical parameters.
        rho_seq : Tensor, shape (B, T, N_RHO)
            Density matrix time series.

        Returns
        -------
        Tensor, shape (B,)
            Log-probability of each parameter vector under the flow posterior.
        """
        context  = self.encoder(rho_seq)
        return self.flow.log_prob(params_norm, context=context)

    def sample_posterior(self, rho_seq: Tensor, n_samples: int = 1000) -> Tensor:
        """
        Draw samples from the posterior P(params | rho_seq).

        Parameters
        ----------
        rho_seq : Tensor, shape (1, T, N_RHO)  or  (B, T, N_RHO)
            Observed trajectory (can include noise).
        n_samples : int
            Number of posterior samples per trajectory.

        Returns
        -------
        Tensor, shape (B, n_samples, n_params)
            Posterior samples in normalised parameter space.
            Use ParamNormaliser.inverse_transform() for physical units.
        """
        B       = rho_seq.shape[0]
        context = self.encoder(rho_seq)               # (B, encoder_out)
        # Repeat context for n_samples
        ctx_rep = context.unsqueeze(1).expand(-1, n_samples, -1)  # (B, n_samples, C)
        ctx_flat = ctx_rep.reshape(B * n_samples, self.encoder_out)

        with torch.no_grad():
            samples = self.flow.sample(1, context=ctx_flat)  # (B*n_samples, 1, n_params)
        # nflows.sample returns shape (N, n_samples_arg, n_params)
        samples = samples.squeeze(1).reshape(B, n_samples, self.n_params)
        return samples

    def posterior_mean(self, rho_seq: Tensor, n_samples: int = 500) -> Tensor:
        """
        Posterior mean (MMSE estimate) via Monte Carlo averaging.

        This is E[theta | rho(t)] — the mean of the posterior distribution.
        It minimises expected squared error, but is NOT the MAP (mode).
        For symmetric unimodal posteriors these coincide; for multimodal
        posteriors they can differ substantially.

        Parameters
        ----------
        rho_seq : Tensor, shape (B, T, N_RHO)
        n_samples : int
            Number of posterior samples to average over.

        Returns
        -------
        Tensor, shape (B, n_params)
            Posterior mean in normalised parameter space.
        """
        samples = self.sample_posterior(rho_seq, n_samples=n_samples)  # (B, N, n_params)
        return samples.mean(dim=1)

    def map_estimate(self, rho_seq: Tensor,
                     n_init_samples: int = 200,
                     n_steps: int = 200,
                     lr: float = 5e-3) -> Tensor:
        """
        Maximum A Posteriori (MAP) estimate via gradient ascent on log P(theta|context).

        Strategy:
          1. Draw n_init_samples from the flow to find a good starting point.
          2. Initialise theta at the highest-log-prob sample (warm start).
          3. Run gradient ascent: theta ← theta + lr * grad_theta log P(theta|c).

        This finds the mode of the posterior, which is the true MAP estimate.
        It is more expensive than posterior_mean but statistically correct.

        Parameters
        ----------
        rho_seq : Tensor, shape (B, T, N_RHO)
        n_init_samples : int
            Samples used to warm-start the optimisation.
        n_steps : int
            Gradient ascent steps.
        lr : float
            Step size.

        Returns
        -------
        Tensor, shape (B, n_params)
            MAP estimate in normalised parameter space.
        """
        self.eval()
        B = rho_seq.shape[0]

        with torch.no_grad():
            context = self.encoder(rho_seq)   # (B, C)

        # Warm start: find the sample with highest log prob per batch element
        init_samples = self.sample_posterior(rho_seq, n_samples=n_init_samples)  # (B, N, D)
        ctx_rep = context.unsqueeze(1).expand(-1, n_init_samples, -1)            # (B, N, C)
        ctx_flat = ctx_rep.reshape(B * n_init_samples, self.encoder_out)
        samp_flat = init_samples.reshape(B * n_init_samples, self.n_params)

        with torch.no_grad():
            lp_flat = self.flow.log_prob(samp_flat, context=ctx_flat)  # (B*N,)
        lp_mat  = lp_flat.reshape(B, n_init_samples)
        best_idx = lp_mat.argmax(dim=1)                                # (B,)
        theta    = init_samples[torch.arange(B), best_idx].clone().detach().requires_grad_(True)
        # (B, n_params) — optimisable

        # Gradient ascent on log P(theta | context)
        opt = torch.optim.Adam([theta], lr=lr)
        ctx_fixed = context.detach()

        for _ in range(n_steps):
            opt.zero_grad()
            log_p = self.flow.log_prob(theta, context=ctx_fixed)  # (B,)
            loss  = -log_p.sum()                                    # minimise NLL
            loss.backward()
            opt.step()

        return theta.detach()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_inverse_from_config(cfg: dict) -> InverseFlowModel:
    mc  = cfg["model"]
    imc = mc["inverse"]
    m   = InverseFlowModel(
        n_rho          = mc["n_rho_elements"],
        n_params       = mc["n_params"],
        encoder_hidden = imc["encoder_hidden"],
        encoder_layers = imc["encoder_layers"],
        encoder_out    = imc["encoder_out"],
        flow_hidden    = imc["flow_hidden"],
        flow_layers    = imc["flow_layers"],
        flow_dropout   = imc["flow_dropout"],
    )
    enc_p  = sum(p.numel() for p in m.encoder.parameters() if p.requires_grad)
    flow_p = sum(p.numel() for p in m.flow.parameters()    if p.requires_grad)
    print(f"Inverse model:  encoder={enc_p:,}  flow={flow_p:,}  "
          f"total={m.count_parameters():,} parameters")
    return m
