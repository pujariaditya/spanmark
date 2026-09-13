"""A zero-started, bounded change to one frozen SSL model's layer mixture.

The inherited mixture remains the value path. Only a bias-free frame-wise
projection observes a detached, normalized copy of that mixture.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def exponential_tilt_delta(weights: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Return softmax(log(weights) + delta) - weights without zero cancellation.

    ``weights`` is a normalized vector and ``delta`` ends in its layer axis.
    This form has an exactly zero value, but a nonzero derivative, at delta=0.
    The caller bounds delta, keeping the denominator safely away from zero.
    """
    if weights.ndim != 1 or delta.shape[-1] != weights.numel():
        raise ValueError("layer weights and residual logits have different shapes")
    scaled = torch.expm1(delta)
    average = (weights * scaled).sum(dim=-1, keepdim=True)
    return weights * (scaled - average) / (1.0 + average)


class ConditionalLayerGate(nn.Module):
    """One trainable matrix, with no construction-time RNG consumption."""

    def __init__(self, input_dim: int, n_layers: int, logit_bound: float = 1.0):
        super().__init__()
        if input_dim < 1 or n_layers < 2:
            raise ValueError("conditional gate requires positive width and multiple layers")
        if not math.isfinite(logit_bound) or not 0 < logit_bound <= 1.0:
            raise ValueError("conditional residual bound must be in (0, 1]")
        self.input_dim = int(input_dim)
        self.n_layers = int(n_layers)
        self.logit_bound = float(logit_bound)
        self.norm_eps = 1e-5
        # Allocating zeros, rather than constructing and reinitializing a Linear,
        # preserves both CPU and CUDA RNG states and the control batch schedule.
        self.weight = nn.Parameter(torch.zeros(self.n_layers, self.input_dim))
        self.last_delta: torch.Tensor | None = None
        self.last_weight_delta: torch.Tensor | None = None

    def forward(
        self, states: torch.Tensor, layer_logits: torch.Tensor,
        inherited: torch.Tensor,
    ) -> torch.Tensor:
        if (states.ndim != 4 or states.shape[0] != self.n_layers
                or states.shape[-1] != self.input_dim
                or inherited.shape != states.shape[1:]
                or inherited.dtype != states.dtype
                or layer_logits.shape != (self.n_layers,)):
            raise ValueError("conditional gate received a different SSL mixture contract")
        if self.weight.dtype != torch.float32:
            raise ValueError("conditional gate parameters must remain FP32")
        with torch.autocast(device_type=states.device.type, enabled=False):
            context = F.layer_norm(inherited.detach().float(), (self.input_dim,),
                                   eps=self.norm_eps)
            delta = self.logit_bound * torch.tanh(F.linear(context, self.weight))
            weights = layer_logits.detach().float().softmax(dim=0)
            adjustment = exponential_tilt_delta(weights, delta)
            # Match the inherited value dtype; do not promote the full L*B*T*D
            # stack to FP32 merely because gate logits are computed in FP32.
            value_adjustment = adjustment.permute(2, 0, 1).unsqueeze(-1).to(states.dtype)
            correction = (states * value_adjustment).sum(dim=0)
        self.last_delta = delta.detach()
        self.last_weight_delta = adjustment.detach()
        return inherited + correction

    def diagnostics(self, n_segments: torch.Tensor) -> dict:
        """Training-only movement evidence over valid frames, never score rules."""
        if self.last_delta is None or self.last_weight_delta is None:
            raise RuntimeError("run the gate before requesting diagnostics")
        delta = self.last_delta
        lengths = n_segments.to(delta.device).clamp(max=delta.shape[1])
        valid = torch.arange(delta.shape[1], device=delta.device)[None, :] < lengths[:, None]
        selected = delta[valid]
        if not selected.numel():
            raise ValueError("gate diagnostics contain no valid frames")
        return {
            "frames": int(valid.sum()),
            "residual_logit_mean_by_layer": selected.mean(0).cpu().tolist(),
            "residual_logit_std_by_layer": selected.std(0, unbiased=False).cpu().tolist(),
            "residual_logit_abs_max": float(selected.abs().max()),
            "weight_delta_l1_mean": float(self.last_weight_delta[valid].abs().sum(-1).mean()),
            "gate_weight_l2": float(self.weight.detach().norm()),
        }
