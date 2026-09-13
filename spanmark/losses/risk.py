"""Training-only binary ranking surrogates for a native bonafide score."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class _SmoothEER(torch.autograd.Function):
    """Implicit gradient of the equal-soft-error crossover, training only."""

    @staticmethod
    def forward(ctx, scores, labels, temperature):
        if scores.ndim != 1 or labels.shape != scores.shape:
            raise ValueError("smooth EER needs aligned rank-one scores and labels")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("smooth EER temperature must be positive")
        # The scalar root has no learned/deployed threshold. Solve in CPU
        # float64: these tiny vectors avoid dozens of GPU synchronization calls.
        values = scores.detach().to(device="cpu", dtype=torch.float64)
        truth = labels.detach().to(device="cpu")
        if not torch.isfinite(values).all() or not torch.all((truth == 0) | (truth == 1)):
            raise ValueError("smooth EER expects finite scores and binary labels")
        positive, negative = truth == 1, truth == 0
        gradient = torch.zeros_like(values)
        if not positive.any() or not negative.any():
            ctx.save_for_backward(gradient.to(scores))
            return scores.new_zeros(())
        bona, spoof = values[positive], values[negative]
        low = values.min() - 32 * temperature
        high = values.max() + 32 * temperature
        for _ in range(56):
            threshold = (low + high) / 2
            bona_error = torch.sigmoid((threshold - bona) / temperature)
            spoof_error = torch.sigmoid((spoof - threshold) / temperature)
            if bona_error.mean() < spoof_error.mean():
                low = threshold
            else:
                high = threshold
        threshold = (low + high) / 2
        bona_error = torch.sigmoid((threshold - bona) / temperature)
        spoof_error = torch.sigmoid((spoof - threshold) / temperature)
        bona_slope = bona_error * (1 - bona_error) / temperature
        spoof_slope = spoof_error * (1 - spoof_error) / temperature
        a, b = bona_slope.mean(), spoof_slope.mean()
        denominator = (a + b).clamp_min(torch.finfo(torch.float64).tiny)
        gradient[positive] = -(b / denominator) * bona_slope / bona.numel()
        gradient[negative] = (a / denominator) * spoof_slope / spoof.numel()
        ctx.save_for_backward(gradient.to(scores))
        return ((bona_error.mean() + spoof_error.mean()) / 2).to(scores)

    @staticmethod
    def backward(ctx, output_gradient):
        gradient, = ctx.saved_tensors
        return output_gradient * gradient, None, None


def smooth_eer_loss(scores: torch.Tensor, labels: torch.Tensor, *, temperature: float = 1.0):
    """Equal soft class-error rate with its implicit, shift-invariant gradient.

    At the solved crossover, a=d(FNR)/dt and b=-d(FPR)/dt. Implicit
    differentiation gives (b*d(FNR)+a*d(FPR))/(a+b). This is a smoothed
    training surrogate, not the grader's discrete EER. A one-class batch
    contributes zero; callers should retain BCE as an anchor.
    """
    return _SmoothEER.apply(scores, labels, float(temperature))


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    tail_fraction: float = 1.0,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft pairwise margin on all pairs or symmetric difficult class tails.

    Larger scores mean bonafide. For q<1, use the lowest q bonafide scores
    and highest q spoof scores. This is an experimental symmetric tail risk,
    not an exact EER loss or the standard one-sided partial AUC objective.
    Selection indices are discrete, while selected scores keep gradients.
    """
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("ranking needs aligned rank-one scores and labels")
    if not 0 < tail_fraction <= 1 or temperature <= 0 or margin < 0:
        raise ValueError("invalid tail fraction, temperature, or margin")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("ranking labels must be 0=spoof or 1=bonafide")
    positive = scores[labels == 1].float()
    negative = scores[labels == 0].float()
    if not positive.numel() or not negative.numel():
        return scores.float().sum() * 0.0
    if tail_fraction < 1:
        positive = positive.topk(
            max(1, math.ceil(positive.numel() * tail_fraction)), largest=False
        ).values
        negative = negative.topk(
            max(1, math.ceil(negative.numel() * tail_fraction)), largest=True
        ).values
    violation = margin - (positive[:, None] - negative[None, :])
    return temperature * F.softplus(violation / temperature).mean()


def native_risk_loss(scores: torch.Tensor, labels: torch.Tensor, arm: dict):
    """BCE anchor plus a separately audited ranking term."""
    bce = F.binary_cross_entropy_with_logits(scores.float(), labels.float())
    rank = pairwise_ranking_loss(
        scores, labels, tail_fraction=arm["tail_fraction"],
        margin=arm["margin"], temperature=arm["temperature"],
    ) if arm["ranking_weight"] else scores.sum() * 0.0
    return bce + arm["ranking_weight"] * rank, bce.detach(), rank.detach()
