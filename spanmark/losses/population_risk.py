"""Training-only CVaR ranking and a scalar population-quantile tracker.

Larger scores mean bonafide. The common-threshold objective is a derived
reparameterization of the monotone pairwise CVaR objective, not literal SOPA.
These functions neither mutate model/state inputs nor define inference outputs.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _beta(value) -> float:
    result = _number(value, "beta")
    if not 0 < result <= 1:
        raise ValueError("beta must lie in (0, 1]")
    return result


def _vector(scores: torch.Tensor, name: str) -> torch.Tensor:
    if (not isinstance(scores, torch.Tensor) or scores.ndim != 1
            or scores.numel() == 0 or not scores.is_floating_point()):
        raise ValueError(f"{name} must be a nonempty floating-point rank-one tensor")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError(f"{name} contains non-finite scores")
    # Tensor.to preserves the original scores' autograd path.
    return scores.to(dtype=torch.float64)


def _scalar(value, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point scalar tensor")
        result = value.to(device=device, dtype=torch.float64)
    else:
        result = torch.tensor(_number(value, name), device=device, dtype=torch.float64)
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _loss_inputs(positive, negative, beta, margin, temperature):
    positive = _vector(positive, "positive")
    negative = _vector(negative, "negative")
    if positive.device != negative.device:
        raise ValueError("positive and negative scores must be on the same device")
    beta = _beta(beta)
    margin = _number(margin, "margin")
    temperature = _number(temperature, "temperature")
    if margin < 0 or temperature <= 0:
        raise ValueError("margin must be nonnegative and temperature must be positive")
    return positive, negative, beta, margin, temperature


def _pair_loss(positive, negative, margin, temperature):
    violation = (margin - positive + negative) / temperature
    # logaddexp evaluates softplus stably without the default softplus
    # float32-oriented linear cutoff, retaining float64 value/gradient accuracy.
    return temperature * torch.logaddexp(torch.zeros_like(violation), violation)


def _finite_loss(loss):
    if not bool(torch.isfinite(loss)):
        raise ValueError("CVaR arithmetic produced a non-finite loss")
    return loss


def empirical_cvar_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    beta: float = 0.1,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact fractional empirical upper-tail CVaR, averaged over all positives.

    For z=beta*M, the largest floor(z) negative scores have weight one;
    the next has weight z-floor(z). Division is by z, including when z<1.
    Sorting retains score gradients, so the boundary negative receives its
    fractional gradient. Stable sorting chooses a deterministic tie subgradient.
    All loss arithmetic and the scalar result are float64.
    """
    positive, negative, beta, margin, temperature = _loss_inputs(
        positive, negative, beta, margin, temperature)
    ordered = torch.sort(negative, descending=True, stable=True).values
    mass = beta * negative.numel()
    whole = math.floor(mass)
    fraction = mass - whole
    selected = whole + int(fraction > 0)
    losses = _pair_loss(positive[:, None], ordered[None, :selected], margin, temperature)
    weights = losses.new_ones(selected)
    if fraction > 0:
        weights[-1] = fraction
    return _finite_loss(((losses * weights).sum(dim=1) / mass).mean())


def population_cvar_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    q,
    *,
    beta: float = 0.1,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CVaR at a common scalar negative-score threshold, with full eta gradients.

    eta_i=L(positive_i,q); return mean(eta)+mean(relu(L_ij-eta_i))/beta.
    In particular eta is never detached: its positive-score gradient supplies
    the correction away from an exact population quantile. A tensor q retains
    its autograd path for derivative checks; training passes a detached q.
    ReLU uses its zero subgradient at an exact tie. q is an independent state
    variable here, unlike a detached order statistic used as an empirical tail.
    """
    positive, negative, beta, margin, temperature = _loss_inputs(
        positive, negative, beta, margin, temperature)
    threshold = _scalar(q, "q", positive.device)
    eta = _pair_loss(positive, threshold, margin, temperature)
    pairs = _pair_loss(positive[:, None], negative[None, :], margin, temperature)
    return _finite_loss(eta.mean() + F.relu(pairs - eta[:, None]).mean() / beta)


def quantile_update(
    q,
    negative: torch.Tensor,
    *,
    beta: float = 0.1,
    gamma: float,
) -> tuple[torch.Tensor, dict]:
    """Return q+gamma*(mean(negative>q)-beta), using old detached values.

    q_next is a new detached scalar float64 tensor on negative.device. Nothing
    is updated in place, and no graph is retained. Call exactly once after the
    native backward; this function applies no clipping, scaling or schedule.
    The JSON-safe diagnostics describe the values used for this actual step.
    """
    beta = _beta(beta)
    gamma = _number(gamma, "gamma")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    with torch.no_grad():
        values = _vector(negative, "negative").detach()
        old = _scalar(q, "q", values.device).detach()
        tail_fraction = (values > old).to(dtype=torch.float64).mean()
        step = gamma * (tail_fraction - beta)
        new = old + step
        if not bool(torch.isfinite(new)):
            raise ValueError("quantile update produced a non-finite threshold")
        diagnostics = {
            "q_before": float(old), "q_after": float(new),
            "r": float(tail_fraction), "step": float(step),
            "gamma": gamma, "beta": beta, "negative_count": values.numel(),
        }
    return new, diagnostics


def threshold_initialization(
    negative_scores,
    *,
    beta: float = 0.1,
    timescale: int = 32,
) -> dict:
    """Registered inverted-CDF q85/q90/q95 and fixed gamma, without updates.

    Every negative population occurrence must be supplied, including repeats.
    Ascending index ceil(p*N)-1 defines each percentile; no interpolation or
    deduplication is performed. Only beta=.1 is supported for this registered
    fixed percentile band. Returned values are JSON-safe Python scalars.
    """
    beta = _beta(beta)
    if beta != 0.1:
        raise ValueError("the registered q85/q90/q95 initialization requires beta=0.1")
    if isinstance(timescale, bool) or not isinstance(timescale, int) or timescale < 1:
        raise ValueError("timescale must be a positive integer update count")
    if isinstance(negative_scores, torch.Tensor):
        values = _vector(negative_scores.detach(), "negative_scores").cpu().numpy()
    else:
        values = np.asarray(negative_scores)
        if (values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.number)
                or np.iscomplexobj(values)):
            raise ValueError("negative_scores must be a nonempty real rank-one array")
    ordered = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(ordered).all():
        raise ValueError("negative_scores contains non-finite scores")
    ordered.sort()
    count = int(ordered.size)
    indices = [max(0, min(count - 1, math.ceil(probability * count) - 1))
               for probability in (0.85, 0.9, 0.95)]
    q85, q90, q95 = (float(ordered[index]) for index in indices)
    gamma = (q95 - q85) / (beta * timescale)
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("registered negative quantile band produced nonpositive or non-finite gamma")
    return {"q85": q85, "q90": q90, "q95": q95, "q0": q90, "gamma": gamma,
            "negative_count": count, "beta": beta, "timescale": timescale,
            "quantile_method": "inverted_cdf", "quantile_indices": indices}
