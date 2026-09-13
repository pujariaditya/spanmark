"""Training-only, two-backward SAM with an exact parameter restore."""
from __future__ import annotations

import math
import torch


def sam_step(parameters, optimizer, closure, *, rho: float, clip: float = 5.0):
    """Take one base-optimizer step using the gradient at an L2 adversary.

    ``closure`` returns a scalar loss and detached diagnostics. Both passes
    must use the same data and deterministic model. No averaged weights or
    inference transformation are produced. Rho zero is ordinary optimization.
    """
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("SAM radius must be finite and nonnegative")
    parameters = tuple(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = closure()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite original loss")
    loss.backward()
    gradients = [p.grad for p in parameters]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("missing or non-finite original gradient")
    norm = torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt()
    adversarial_loss = loss.detach()
    actual_radius = 0.0
    if rho:
        originals = [p.detach().clone() for p in parameters]
        try:
            with torch.no_grad():
                scale = rho / norm.clamp_min(1e-12)
                for p, g in zip(parameters, gradients):
                    p.add_(g * scale)
                if diagnostics.get("audit_radius", False):
                    actual_radius = float(torch.stack([
                        (p - old).square().sum() for p, old in zip(parameters, originals)
                    ]).sum().sqrt())
            optimizer.zero_grad(set_to_none=True)
            perturbed, _ = closure()
            if not bool(torch.isfinite(perturbed)):
                raise RuntimeError("non-finite perturbed loss")
            perturbed.backward()
            adversarial_loss = perturbed.detach()
        finally:
            # Subtracting the perturbation can accumulate FP32 rounding drift.
            with torch.no_grad():
                for p, old in zip(parameters, originals):
                    p.copy_(old)
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise RuntimeError("missing or non-finite adversarial gradient")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return dict(diagnostics, loss=float(loss.detach()),
                adversarial_loss=float(adversarial_loss), original_gradient_norm=float(norm),
                update_gradient_norm=float(grad_norm), actual_radius=actual_radius)
